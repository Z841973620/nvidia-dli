# Import necessary GStreamer libraries and DeepStream python bindings
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from common.bus_call import bus_call

def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create Pipeline element that will form a connection of other elements
    pipeline=Gst.Pipeline()
    print("Created pipeline")

    # Create Source element for reading from a file and set the location property
    source=Gst.ElementFactory.make("filesrc", "file-source")
    source.set_property('location', args[1])

    # Create H264 Parser with h264parse as the input file is an elementary h264 stream
    h264parser=Gst.ElementFactory.make("h264parse", "h264-parser")

    # Create Decoder with nvv4l2decoder for accelerated decoding on GPU
    decoder=Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")

    # Create Streamux with nvstreammux to form batches for one or more sources and set properties
    streammux=Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 888)
    streammux.set_property('height', 696)
    streammux.set_property('batch-size', 1)

    # Create Primary GStreamer Inference Element with nvinfer to run inference on the decoder's output after batching
    pgie=Gst.ElementFactory.make("nvinfer", "primary-inference")
    # Behaviour of inferencing is set through config file
    pgie.set_property('config-file-path', '/dli/task/sample_apps/pgie_config_trafficcamnet_02.txt')

    # Create Convertor to convert from YUV to RGBA as required by nvosd
    nvvidconv1=Gst.ElementFactory.make("nvvideoconvert", "convertor")

    # Create OSD with nvdsosd to draw on the converted RGBA buffer
    nvosd=Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # Create Convertor to convert from RGBA to I420 as required by encoder
    nvvidconv2=Gst.ElementFactory.make("nvvideoconvert", "convertor2")

    # Create Capsfilter to enforce frame image format
    capsfilter=Gst.ElementFactory.make("capsfilter", "capsfilter")
    caps=Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    # Create Encoder to encode I420 formatted frames using the MPEG4 codec
    encoder=Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    encoder.set_property("bitrate", 2000000)
    
    # Create Sink and set the location for the output file
    sink=Gst.ElementFactory.make('filesink', 'filesink')
    sink.set_property('location', '/dli/task/output_02_encoded.mpeg4')
    sink.set_property('sync', 1)
    print('Created elements')
    
    # Add elements to pipeline
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(sink)
    print("Added elements to pipeline")

    # Link elements in the pipeline
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> nvvidconv -> capsfilter -> encoder -> sink
    source.link(h264parser)
    h264parser.link(decoder)

    decoder_srcpad=decoder.get_static_pad("src")
    streammux_sinkpad=streammux.get_request_pad("sink_0")
    decoder_srcpad.link(streammux_sinkpad)
    
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(sink)
    print('Linked elements in pipeline')
    
    # Create an event loop
    loop=GLib.MainLoop()
    
    # Feed GStreamer bus mesages to loop
    bus=pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    print('Added bus message handler')

    # Start play back and listen to events
    print("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    
    # Cleaning up as the pipeline comes to an end
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))