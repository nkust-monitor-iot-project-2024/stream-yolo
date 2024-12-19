use anyhow::Context;
use glib::object::Cast;
use gst::prelude::*;
use gstreamer as gst;
use gstreamer::prelude::ElementExt;
use gstreamer_app::AppSinkCallbacks;
use gstreamer_video as gst_video;
use image::{DynamicImage, ImageFormat, ImageReader, RgbImage};
use std::fs::File;
use std::{
    env,
    sync::atomic::{AtomicUsize, Ordering},
};
use yolo_rs::model::YoloModelSession;
use yolo_rs::{BoundingBox, YoloInput, image_to_yolo_input_tensor, inference};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Initialize GStreamer
    gst::init()?;

    // Check for RTSP stream URI argument
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <RTSP URL>", args[0]);
        return Ok(());
    }
    let rtsp_url = &args[1];

    let pipeline = gstreamer::Pipeline::new();

    let rtspsrc_element = gst::ElementFactory::make("rtspsrc")
        .property("location", rtsp_url)
        .build()
        .context("failed to create rtspsrc element")?;

    let rtpjitterbuffer_element = gst::ElementFactory::make("rtpjitterbuffer")
        .build()
        .context("failed to create rtpjitterbuffer element")?;

    let rtph264depay_element = gst::ElementFactory::make("rtph264depay")
        .property("wait-for-keyframe", true)
        .property("request-keyframe", true)
        .build()
        .context("failed to create rtph264depay element")?;

    let avdec_h264_element = gst::ElementFactory::make("avdec_h264")
        .build()
        .context("failed to create avdec_h264 element")?;

    let videoconvert_element = gst::ElementFactory::make("videoconvert")
        .build()
        .context("failed to create videoconvert element")?;

    let identity_element = gst::ElementFactory::make("identity")
        .property("check-imperfect-offset", true)
        .property("check-imperfect-timestamp", true)
        .build()
        .context("failed to create identity element")?;

    let frame_counter = AtomicUsize::new(0);

    let yolo_model = YoloModelSession::from_filename_v8(
        "/Volumes/Dev/nkust/iot/yolo-v11-rs/examples/yolo-cli/models/yolo11x.onnx",
    )
    .context("failed to load YOLO model")?;

    let appsink_callback = AppSinkCallbacks::builder()
        .new_sample(move |sink| {
            let sample = match sink.pull_sample() {
                Ok(sample) => sample,
                Err(_) => return Err(gst::FlowError::Error),
            };

            // Extract the buffer and caps (metadata)
            let buffer = sample.buffer().unwrap();
            let caps = sample.caps().unwrap();
            let video_info = gst_video::VideoInfo::from_caps(caps).unwrap();

            // Convert the buffer to a readable format
            let map = buffer.map_readable().unwrap();

            // Increment the frame counter
            let counter = frame_counter.fetch_add(1, Ordering::Relaxed);

            // Save frame as PNG every second (assuming 1 frame per second)
            if counter % 30 == 0 {
                // Adjust based on your stream's FPS
                let width = video_info.width() as usize;
                let height = video_info.height() as usize;

                // Extract the frame data
                let frame_data = map.as_slice();

                let frame = RgbImage::from_raw(width as u32, height as u32, frame_data.to_vec())
                    .expect("expect a valid image");
                let dynamic_image = DynamicImage::ImageRgb8(frame);

                tracing::info!("Inferring frame {}", counter);
                let now = std::time::Instant::now();

                let yolo_input = image_to_yolo_input_tensor(&dynamic_image);
                let yolo_output =
                    inference(&yolo_model, yolo_input.view()).expect("failed to run inference");

                tracing::info!(
                    "Found {} entities, elapsed: {:?}",
                    yolo_output.len(),
                    now.elapsed()
                );

                // extract the entity to few pictures
                for entity in yolo_output {
                    let BoundingBox { x1, x2, y1, y2 } = entity.bounding_box;
                    let label = entity.label;
                    let confidence = entity.confidence;

                    let cropped_image = dynamic_image.crop_imm(
                        x1 as _,
                        y1 as _,
                        (x2 - x1) as u32,
                        (y2 - y1) as u32,
                    );

                    // save the image to "frame-<counter>-<label>-<confidence>.png"
                    let mut file =
                        File::create(format!("frame-{}-{}-{:.2}.png", counter, label, confidence))
                            .expect("expect a valid file");
                    cropped_image
                        .write_to(&mut file, ImageFormat::Png)
                        .expect("expect a valid image");
                }
            }

            Ok(gst::FlowSuccess::Ok)
        })
        .build();

    let appsink_element = gstreamer_app::AppSink::builder()
        .name("appsink")
        .sync(true)
        .callbacks(appsink_callback)
        .caps(
            &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .build(),
        )
        .build()
        .upcast();

    pipeline.add_many([
        &rtspsrc_element,
        &rtpjitterbuffer_element,
        &rtph264depay_element,
        &avdec_h264_element,
        &videoconvert_element,
        &identity_element,
        &appsink_element,
    ])?;

    let rtpjitterbuffer_element_clone = rtpjitterbuffer_element.clone();
    rtspsrc_element.connect_pad_added(move |_, src_pad| {
        let sink_pad = rtpjitterbuffer_element_clone.static_pad("sink").unwrap();
        if !sink_pad.is_linked() {
            match src_pad.link(&sink_pad) {
                Ok(_) => tracing::info!("Successfully linked pads"),
                Err(err) => tracing::warn!("Failed to link pads: {:?}", err),
            }
        }
    });

    // link elements
    gst::Element::link_many([
        &rtpjitterbuffer_element,
        &rtph264depay_element,
        &avdec_h264_element,
        &videoconvert_element,
        &identity_element,
        &appsink_element,
    ])?;

    // Start the pipeline
    pipeline
        .set_state(gst::State::Playing)
        .context("failed to start pipeline")?;

    // Wait until error or EOS
    let bus = pipeline.bus().context("failed to get bus")?;
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        match msg.view() {
            gst::MessageView::Eos(..) => break,
            gst::MessageView::Error(err) => {
                eprintln!(
                    "Error from {}: {}",
                    err.src().map(|s| s.path_string()).unwrap_or("<?>".into()),
                    err.error()
                );
                break;
            }
            _ => (),
        }
    }

    // Shutdown pipeline
    pipeline
        .set_state(gst::State::Null)
        .context("failed to stop pipeline")?;

    Ok(())
}
