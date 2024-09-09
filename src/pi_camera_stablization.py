# import required libraries
from vidgear.gears.stabilizer import Stabilizer
from vidgear.gears import CamGear
import cv2

# open stream with default parameters
stream = PiGear().start()

# initiate stabilizer object with default parameters
stab = Stabilizer()

outputPipeline = "appsrc ! videoconvert ! capsfilter caps=\"video/x-raw,format=I420,width=1280,height=720,framerate=30/1\" ! x264enc ! rtph264pay config-interval=1 pt=96 ! udpsink host=\""+host+"\" port="+port+" sync=false";

out=cv2.VideoWriter(outputPipeline, cv2.CAP_GSTREAMER, 0, float(30), (int(1280), int(720)))

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # send current frame to stabilizer for processing
    stabilized_frame = stab.stabilize(frame)

    # wait for stabilizer which still be initializing
    if stabilized_frame is None:
        continue

    out.write(stabilized_frame);

    # {do something with the stabilized frame here}

    # Show output window
    # cv2.imshow("Output Stabilized Frame", stabilized_frame)

    # check for 'q' key if pressed
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord("q"):
    #    break

# close output window
#//cv2.destroyAllWindows()

# clear stabilizer resources
stab.clean()

# safely close video stream
stream.stop()