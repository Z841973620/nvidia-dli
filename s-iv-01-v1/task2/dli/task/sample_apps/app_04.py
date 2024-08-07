#Import necessary libraries
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from common.bus_call import bus_call
import pyds
import time
import math

MAX_DISPLAY_LEN=64
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
start=time.time()

def main(args):
    number_sources=int(args[3])

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline")
    pipeline = Gst.Pipeline()

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property('width', 888)
    streammux.set_property('height', 696)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    
    pipeline.add(streammux)
    
    ########################
    # -=START EDITING HERE=-
    
    # args[1] <path to input h264 video> 
    # args[2] <path to nvinfer config file> 
    # args[3] <number of file sources> 
    # args[4] <name of output file>
    
    # Iteratively create filesrc -> h264pase -> nvv4l2decoder.get_static_pad('src') -> streammux.get_request_pad('sink_%u')
    for i in range(number_sources): 
        print('Creating source_bin ', i, end='\r')
        source=Gst.ElementFactory.make('filesrc', 'file-source_%u'%i)
        source.set_property('location', args[1])
        h264parser=Gst.ElementFactory.make('h264parse', 'h264-parser_%u'%i)
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder_%u"%i)
        pipeline.add(<<<<FIXME>>>>)
        pipeline.add(<<<<FIXME>>>>)
        pipeline.add(<<<<FIXME>>>>)
        padname="sink_%u"%i
        <<<<FIXME>>>>.link(<<<<FIXME>>>>)
        <<<<FIXME>>>>.link(<<<<FIXME>>>>)
        <<<<FIXME>>>>.get_static_pad("src").link(<<<<FIXME>>>>.get_request_pad(padname))
    
    # -=END EDIT=-
    ########################
    
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', args[2])
    pgie_batch_size=pgie.get_property("batch-size")
    
    # Check if batch-size in nvinfer is same as the number of input sources
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ", number_sources)
        pgie.set_property("batch-size", number_sources)
    
    # Implement a nvmultistreamtiler and set properties related to dimensions
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)
    
    encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    encoder.set_property("bitrate", 2000000)
    
    codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
    container = Gst.ElementFactory.make("qtmux", "qtmux")
    sink=Gst.ElementFactory.make('filesink', 'filesink')
    sink.set_property('location', args[4])
    sink.set_property("sync", 1)
    sink.set_property("async", 0)

    print('Adding elements to Pipeline')
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(sink)
    
    print("Linking elements in the Pipeline")
    streammux.link(pgie)
    pgie.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(codeparser)
    codeparser.get_static_pad("src").link(container.get_request_pad("video_0"))
    container.link(sink)
    
    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    tiler_src_pad=pgie.get_static_pad("src")
    tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    for i in range(number_sources+1): 
        if (i != 0):
            print(i, ": ", args[1])

    print("Starting pipeline")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    start_time=time.time()
    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)
    print("--- %s seconds ---" % (time.time() - start_time))

# tiler_sink_pad_buffer_probe will extract metadata received on OSD sink pad and update params for drawing rectangle, object information etc.
def tiler_src_pad_buffer_probe(pad, info, u_data):
    global start
    frame_number=0
    total_frames=0
    now=time.time()
    num_rects=0
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        total_frames+=1    
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number=frame_meta.frame_num
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    now=time.time()
    if frame_number%100==0: 
        print(f'FPS: {round(total_frames/(now-start), 2)} @ Frame {frame_number}.')
    start=now
    return Gst.PadProbeReturn.OK
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))