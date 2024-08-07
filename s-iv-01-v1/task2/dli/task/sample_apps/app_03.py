#Import necessary libraries
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from common.bus_call import bus_call
import pyds
import time

start=time.time()

def main(args):
    Gst.init(None)

    # Create element that will form a pipeline
    print("Creating Pipeline")
    pipeline = Gst.Pipeline()
    
    source = Gst.ElementFactory.make("filesrc", "file-source")
    source.set_property('location', args[1])
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")    
    streammux.set_property('width', 888)
    streammux.set_property('height', 696)
    streammux.set_property('batch-size', 1)
    
    pgie = Gst.ElementFactory.make('nvinfer', "primary-inference")
    pgie.set_property('config-file-path', args[2])
    
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor")
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
    sink.set_property('location', args[3])
    sink.set_property("sync", 1)
    sink.set_property("async", 0)
    
    # Add the elements to the pipeline
    print("Adding elements to Pipeline")
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
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(sink)

    # Link the elements together
    print("Linking elements in the Pipeline")
    source.link(h264parser)
    h264parser.link(decoder)
    decoder.get_static_pad('src').link(streammux.get_request_pad("sink_0"))
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(codeparser)
    codeparser.get_static_pad("src").link(container.get_request_pad("video_0"))
    container.link(sink)
    
    # Attached probe to osd sink pad
    osdsinkpad=nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe_fps, 0)
    
    # Create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    # Start play back and listen to events
    print("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    start_time=time.time()
    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)
    print("--- %s seconds ---" % (time.time() - start_time))

# Define the Probe Function
def osd_sink_pad_buffer_probe_fps(pad,info,u_data):
    global start
    frame_number=0
    total_frames=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        total_frames+=1
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number=frame_meta.frame_num
        
        # print(f'Frame: {frame_number}.', end='\r')
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