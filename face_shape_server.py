import flask
import werkzeug
import time
import numpy as np
import cv2
from keras import models
from keras.preprocessing import image


app = flask.Flask(__name__)

''' This function returns the predicted face shape of the model '''
def predict_face_shape(img_path, model_path):
    # Based on what the model predicts, it returns the face Shape as a String
    prediction_mapping = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

    # Load and preprocess the image for the model
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    imgs = np.vstack([img_tensor])

    # Load the model and predict
    loaded_model = models.load_model(model_path)
    prediction = loaded_model.predict(imgs)
    pred = list(prediction[0])
    index = pred.index(max(pred))
    confidence = round(max(pred)*100, 2)

    return prediction_mapping[index], confidence

''' This function detects the are of the face in an image to identify '''
def highlightFace(net, frame, conf_threshold=0.7):
    # Create shallow copy of frame
    frameOpencvDnn=frame.copy()
    # Get height and width
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]

    # Create a blob from shallow copy
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # Set the input
    net.setInput(blob)
    # Make a forward pass to the network
    detections=net.forward()
    faceBoxes=[]
    # For each value in 0-127
    for i in range(detections.shape[2]):
        # Define confidence between 0-1
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            # Get x1, y1, x2, y2 coordinates
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            # Append a list to faceBoxes
            faceBoxes.append([x1,y1,x2,y2])
            # Put the rectangles on the image for each such list of coordinates
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

''' Returns predicted gender and estimated age'''
def predict_age_gender(frame, padding=20):
    # Protocol buffer and model for face, age, and gender
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    # Mean values for the model
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load the network.
    # Pass trained weights and network configuration
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    if not faceBoxes:
        return None, None

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1),
                    max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        prediction = f"\nGender: {gender}\nEstimated Age: {age}"

    return resultImg, prediction

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    model_path = "faceshapeone_vgg16_augment.h5"

    files_ids = list(flask.request.files)
    image_num = 1
    predicted_face_shape = "NO FACE"
    confidence = 1_000_000_000_0000

    for file_id in files_ids:
        imagefile = flask.request.files[file_id]
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        img_path = timestr + '-' + filename
        imagefile.save(img_path)

        image_num = image_num + 1

        predicted_face_shape, confidence = predict_face_shape(img_path, model_path)
        input_img = cv2.imread(img_path)
        result_img, predicted_age_gender = predict_age_gender(input_img)

        if predicted_age_gender == None:
            predicted_age_gender = "\nNo face detected"
            predictions = predicted_age_gender
        else:
            #predictions = str(predicted_face_shape + "\nConfidence Level: " + str(confidence) + "%" )
            predictions = str(predicted_face_shape)
            predictions += predicted_age_gender


        if image_num > 1:
            break

    return predictions

app.run(host="0.0.0.0", port=5000, debug=True)


'''
if __name__ == '__main__':
    # Put location of the file path
    model_path = "faceshapeone_vgg16_augment.h5"

    filenames = ["round_" + str(i) + ".jpg" for i in range(1, 4)]
    print("Testing Round Face:")
    for filename in filenames:
        predicted_face_shape = predict_face_shape(filename, model_path)
        print("Predicted Face Shape: " + predicted_face_shape)

    filenames = ["heart_" + str(i) + ".jpg" for i in range(1, 4)]
    print("Testing Heart Face:")
    for filename in filenames:
        predicted_face_shape = predict_face_shape(filename, model_path)
        print("Predicted Face Shape: " + predicted_face_shape)

    filenames = ["oblong_" + str(i) + ".jpg" for i in range(1, 4)]
    print("Testing Oblong Face:")
    for filename in filenames:
        predicted_face_shape = predict_face_shape(filename, model_path)
        print("Predicted Face Shape: " + predicted_face_shape)

    filenames = ["square_" + str(i) + ".jpg" for i in range(1, 4)]
    print("Testing square Face:")
    for filename in filenames:
        predicted_face_shape = predict_face_shape(filename, model_path)
        print("Predicted Face Shape: " + predicted_face_shape)

    filenames = ["oval_" + str(i) + ".jpg" for i in range(1, 4)]
    print("Testing oval Face:")
    for filename in filenames:
        predicted_face_shape = predict_face_shape(filename, model_path)
        print("Predicted Face Shape: " + predicted_face_shape)
'''



