# A Redis gear for orchestrating realtime video analytics
import cv2
import redisAI
import numpy as np
from PIL import Image
from PIL import ImageDraw
# Configuration
FPS = 10.0          # Maximum number of frames per second to process TODO: move to config key

# Globals for downsampling
_mspf = 1000 / FPS  # Msecs per frame
_next_ts = 0        # Next timestamp to sample

def log(s):
    execute('DEBUG', 'LOG', s)

def downsampleStream(x):
    ''' Drops input frames to match FPS '''
    global _mspf, _next_ts

    execute('TS.INCRBY', 'camera:0:in_fps', 1, 'RESET', 1)  # Store the input fps count
    ts, _ = map(int, str(x['streamId']).split('-'))         # Extract the timestamp part from the message ID
    sample_it = _next_ts <= ts
    if sample_it:                                           # Drop frames until the next timestamp is in the present/past
        _next_ts = ts + _mspf
    return sample_it

def process_image(img, height):
    ''' Utility to resize a rectangular image to a padded square '''
    color = (127.5, 127.5, 127.5)
    shape = img.shape[:2]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (height - new_shape[0]) / 2    # Width padding
    dh = (height - new_shape[1]) / 2    # Height padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    img = np.asarray(img, dtype=np.float32) 
    img /= 255.0                        # Normalize 0..255 to 0..1.00
    return img

def runYolo(x):
    ''' Runs the model on an input image using RedisAI '''
    IMG_SIZE = 416     # Model's input image size

    # Read the image from the stream's message
    buf = io.BytesIO(x['image'])
    pil_image = Image.open(buf)
    numpy_img = np.array(pil_image)
    image = process_image(numpy_img, IMG_SIZE)

    # Prepare the image tensor as model's input (number of tensors, width, height, channels)
    image_tensor = redisAI.createTensorFromBlob('FLOAT', [1, IMG_SIZE, IMG_SIZE, 3], image.tobytes())

    # Create yolo's RedisAI model runner and run it
    modelRunner = redisAI.createModelRunner('yolo:model')
    redisAI.modelRunnerAddInput(modelRunner, 'input', image_tensor)
    redisAI.modelRunnerAddOutput(modelRunner, 'output')
    model_replies = redisAI.modelRunnerRun(modelRunner)
    model_output = model_replies[0]

    # The model's output is processed with a PyTorch script for non maxima suppression
    scriptRunner = redisAI.createScriptRunner('yolo:script', 'boxes_from_tf')
    redisAI.scriptRunnerAddInput(scriptRunner, 'input', model_output)
    redisAI.scriptRunnerAddOutput(scriptRunner, 'output')
    script_reply = redisAI.scriptRunnerRun(scriptRunner)

    # The script's outputs bounding boxes that are serialized
    shape = redisAI.tensorGetDims(script_reply)
    buf = redisAI.tensorGetDataAsBlob(script_reply)
    boxes = np.frombuffer(buf, dtype=np.float32).reshape(shape)

    # # Extract the people boxes
    ratio = float(IMG_SIZE) / max(pil_image.width, pil_image.height)  # ratio = old / new
    pad_x = (IMG_SIZE - pil_image.width * ratio) / 2                  # Width padding
    pad_y = (IMG_SIZE - pil_image.height * ratio) / 2                 # Height padding
    boxes_out = []
    people_count = 0
    for box in boxes[0]:
        if box[4] == 0.0:  # Remove zero-confidence detections
            continue
        if box[-1] != 14:  # Ignore detections that aren't people
            continue
        people_count += 1

        # Descale bounding box coordinates back to original image size
        x1 = (IMG_SIZE * (box[0] - 0.5 * box[2]) - pad_x) / ratio
        y1 = (IMG_SIZE * (box[1] - 0.5 * box[3]) - pad_y) / ratio
        x2 = (IMG_SIZE * (box[0] + 0.5 * box[2]) - pad_x) / ratio
        y2 = (IMG_SIZE * (box[1] + 0.5 * box[3]) - pad_y) / ratio

        # Store boxes as a flat list
        boxes_out += [x1,y1,x2,y2]

    return x['streamId'], people_count, boxes_out

def storeResults(x):
    ''' Stores the results in Redis Stream and TimeSeries data structures '''
    ref_id, people, boxes= x[0], int(x[1]), x[2]
    ref_msec = int(str(ref_id).split('-')[0])

    # Store the output in its own stream
    res_id = execute('XADD', 'camera:0:yolo', 'MAXLEN', '~', 1000, '*', 'ref', ref_id, 'boxes', boxes, 'people', people)

    # Add a sample to the output people and fps timeseries
    res_msec = int(str(res_id).split('-')[0])
    execute('TS.ADD', 'camera:0:people', ref_msec/1000, people)
    execute('TS.INCRBY', 'camera:0:out_fps', 1, 'RESET', 1)

    # Make an arithmophilial homage to Count von Count for storage in the execution log
    if people == 0:
        return 'Now there are none.'
    elif people == 1:
        return 'There is one person in the frame!'
    elif people == 2:
        return 'And now there are are two!'
    else:
        return 'I counted {} people in the frame! Ah ah ah!'.format(people)

# Create and register a gear that for each message in the stream
gb = GearsBuilder('StreamReader')
gb.filter(downsampleStream)  # Filter out high frame rate
gb.map(runYolo)              # Run the model
gb.map(storeResults)         # Store the results
gb.register('camera:0')
