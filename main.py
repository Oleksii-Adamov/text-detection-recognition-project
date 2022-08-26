# imports
import os
import string
import numpy as np
import cv2
import keras_ocr
import timeit

if __name__ == '__main__':
    # changing os directory to directory of this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # defining alphabet
    ukr_letters = 'бвгґдєжзиїйклмнптуфцчшщьюяБГҐДЄЖЗИЇЙЛПУФЦЧШЩЬЮЯ'
    en_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alphabet = string.digits + ukr_letters + en_letters + ' '

    # loading weights
    detector = keras_ocr.detection.Detector(weights='clovaai_general')

    recognizer = keras_ocr.recognition.Recognizer(
        alphabet=alphabet
    )
    recognizer.model.load_weights('./weights/recognizer_2022-08-24T17_19_42.008680.h5')
    pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)

    vid = cv2.VideoCapture(0)

    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # converting to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = timeit.default_timer()
        prediction = pipeline.recognize([frame_rgb])[0]
        print(timeit.default_timer() - start_time)

        for _, box in prediction:
            cv2.polylines(
                img=frame,
                pts=box[np.newaxis].astype("int32"),
                color=(0, 0, 255),
                thickness=1,
                isClosed=True,
            )

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
