function computeColorForLabels(classID){
    if(classID == 0){
        color=[85, 45, 255, 255]
    }
    else if(classID == 2){
        color=[222, 82, 175, 255]
    }
    else if(classID == 3){
        color=[0, 204, 255, 255]
    }
    else if(classID == 4){
        color = [0, 149, 255, 255]
    }
    else{
        color = [255,111,111,255]
    }
    return color;
}
function handleImageInput(event){
    const fileInput = event.target;
    const file = fileInput.files[0];
    if (file){
        const reader = new FileReader();
        reader.onload = function (e) {
            const imgMain = document.getElementById("image-main");
            imgMain.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
}
function downloadImage() {
    // Get the canvas element
    const canvas = document.getElementById('main-canvas');

    // Create an anchor element to trigger the download
    const link = document.createElement('a');

    // Set the download attribute with a filename (you can customize the filename)
    link.download = 'objects_detection.png';

    // Convert the canvas content to a data URL
    const dataUrl = canvas.toDataURL();

    // Set the href attribute of the anchor with the data URL
    link.href = dataUrl;

    // Append the anchor to the document
    document.body.appendChild(link);

    // Trigger a click on the anchor element to start the download
    link.click();

    // Remove the anchor element from the document
    document.body.removeChild(link);
} 
function opencvReady(){
    cv["onRuntimeInitialized"] = () =>
    {
        console.log("OpenCV Ready");
        const labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light","fire hydrant", "stop sign", "parking meter", "bench", "bird","cat", "dog", "horse", "sheep", "cow","elephant", "bear", "zebra", "giraffe", "backpack","umbrella", "handbag", "tie", "suitcase", "frisbee","skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle","wine glass", "cup", "fork", "knife", "spoon","bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot", "hot dog", "pizza", "donut","cake", "chair", "couch", "potted plant", "bed","dining table", "toilet", "tv", "laptop", "mouse","remote", "keyboard", "cell phone", "microwave", "oven","toaster", "sink", "refrigerator", "book", "clock","vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        const numClass = 80;
        const FPS =24;
        //Read an image from the Image Source and Convert it into OpenCV Format
        let imgMain = cv.imread("image-main");
        cv.imshow("main-canvas", imgMain);
        imgMain.delete();
        //Handle Image Input
        document.getElementById("image-upload").addEventListener('change', handleImageInput);
    
        //RGB Image
        document.getElementById("RGB-Image").onclick = function(){
            console.log("RGB Button Pressed");
            let imgMain = cv.imread("image-main");
            cv.imshow("main-canvas", imgMain);
            imgMain.delete();
        }

        //Gray Scale Image
        document.getElementById("Gray-Scale-Image").onclick = function(){
            console.log("Gray Scale Image Button Pressed");
            let imgMain = cv.imread("image-main");
            let imgGray = new cv.Mat();
            cv.cvtColor(imgMain, imgGray, cv.COLOR_RGBA2GRAY);
            cv.imshow("main-canvas", imgGray);
            imgMain.delete();
            imgGray.delete();//Free up the Memory
        }

        //Object Detection Image
        document.getElementById("Object-Detection-Image").onclick = async function(){
            console.log("Object Detection on Image");
            // const labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light","fire hydrant", "stop sign", "parking meter", "bench", "bird","cat", "dog", "horse", "sheep", "cow","elephant", "bear", "zebra", "giraffe", "backpack","umbrella", "handbag", "tie", "suitcase", "frisbee","skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle","wine glass", "cup", "fork", "knife", "spoon","bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot", "hot dog", "pizza", "donut","cake", "chair", "couch", "potted plant", "bed","dining table", "toilet", "tv", "laptop", "mouse","remote", "keyboard", "cell phone", "microwave", "oven","toaster", "sink", "refrigerator", "book", "clock","vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
            // const numClass = 80;
            let image = document.getElementById("image-main");
            let inputImage = cv.imread(image);
            console.log("Input Image Width", inputImage.cols, "Input Image Height", inputImage.rows);
            //Load the TensorFlow js Model
            model = await tf.loadGraphModel("yolov8n_web_model/model.json");
            //Calculate the Model Width and Height
            const inputTensorShape = model.inputs[0].shape;
            const modelWidth = inputTensorShape[1];
            const modelHeight = inputTensorShape[2];
            console.log("Model Width", modelWidth, "Model Height", modelHeight);
            const preprocess = (image, modelWidth, modelHeight) =>{
                let xRatio,  yRatio;
                const input = tf.tidy(()=>{
                    //Convert the Pixel Data From Image Source into TensorFLow js tensor
                    const img = tf.browser.fromPixels(image);
                    //Extracting the Width and Height of Image Tensor
                    const [h,w] = img.shape.slice(0,2);
                    //Height and Width of the Image Tensor
                    console.log("Height", h, "Width", w);
                    //Max Value Between Width and Height
                    const maxSize = Math.max(w, h);
                    //Applying Padding
                    const imgPadded = img.pad([
                        [0, maxSize - h],
                        [0, maxSize - w],
                        [0,0]
                    ]);
                    xRatio = maxSize/w;
                    yRatio = maxSize/h;
                    //Apply Bilinear Interpolation
                    return tf.image.resizeBilinear(imgPadded, [modelWidth, modelHeight]).div(255.0).expandDims(0);
                })
            return [input, xRatio, yRatio]
            };
            const [input, xRatio, yRatio] = preprocess(image, modelWidth, modelHeight);
            console.log("Input Shape", input.shape, "X-Ratio", xRatio, "Y-Ratio", yRatio);

            //Pass Input Image Tensor to the TensorFlow js Model
            const res = model.execute(input);
            //Rearrange the Dimensions of the Tensor
            //[batch_size, height, width] ==> [ batch_size, width, height];
            const transres = res.transpose([0,2,1]);
            const boxes = tf.tidy(()=>{
                const w = transres.slice([0,0,2], [-1,-1,1]);
                const h = transres.slice([0,0,3], [-1,-1,1]);
                const x1 = tf.sub(transres.slice([0,0,0], [-1,-1,1]), tf.div(w,2));
                const y1 = tf.sub(transres.slice([0,0,1], [-1,-1,1]), tf.div(h,2));
                return tf.concat([y1, x1, tf.add(y1, h), tf.add(x1, w)], 2).squeeze();
            });
            //Calcualte the Confidence Score and Class Names
            const [scores, classes] = tf.tidy(() => {
                const rawScores = transres.slice([0,0,4], [-1,-1, numClass]).squeeze(0);
                return [rawScores.max(1), rawScores.argMax(1)];
            });
            //Applying Non Max Supression
            const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500,0.45, 0.2);
            const predictionsLength = nms.size;
            console.log("Predictions Length", predictionsLength)

            if(predictionsLength > 0){
                const boxes_data = boxes.gather(nms, 0).dataSync();
                const score_data = scores.gather(nms, 0).dataSync();
                const classes_data = classes.gather(nms, 0).dataSync();
                console.log("Boxes Data", boxes_data, "Score Data", score_data, "Classes Data", classes_data);
                const xScale = inputImage.cols/modelWidth;
                const yScale = inputImage.rows/ modelHeight;
                console.log("Score Data Length", score_data.length);
                for (let i=0; i < score_data.length; ++i){
                    const classID = classes_data[i];
                    const className = labels[classes_data[i]];
                    const ConfidenceScore = (score_data[i] * 100).toFixed(1);
                    let [y1, x1, y2, x2] = boxes_data.slice(i*4, (i+1)*4);
                    x1 *= xRatio * xScale;
                    x2 *= xRatio * xScale;
                    y1 *= yRatio * yScale;
                    y2 *= yRatio * yScale;
                    const height = y2 - y1;
                    const width = x2 - x1;
                    console.log(x1, y1, width, height, className, ConfidenceScore);
                    let point1 = new cv.Point(x1, y1);
                    let point2 = new cv.Point(x1+ width, y1 + height);
                    cv.rectangle(inputImage, point1, point2, computeColorForLabels(classID), 4);
                    //const text = `${className} - ${Math.round(confidenceScore)/100}`
                    const text = className + " - " + ConfidenceScore + "%"
                    // Create a hidden canvas element to measure the text size
                    const canvas = document.createElement("canvas");
                    const context = canvas.getContext("2d");
                    context.font = "22px Arial"; // Set the font size and family as needed
                    // Measure the width of the text
                    const textWidth = context.measureText(text).width;
                    cv.rectangle(inputImage, new cv.Point(x1,y1-20), new cv.Point(x1+ textWidth + context.lineWidth, y1), computeColorForLabels(classID),-1)
                    cv.putText(inputImage, text, new cv.Point(x1, y1 - 5),cv.FONT_HERSHEY_TRIPLEX, 0.50, new cv.Scalar(255,255,255,255), 1);
                 }
               cv.imshow("main-canvas", inputImage);  

            }
        else{
            cv.imshow("main-canvas", inputImage);
        }
        tf.dispose([res, transres, boxes, scores, classes, nms]);
        }
        //Download Image
        document.getElementById("Download-Image").addEventListener('click', downloadImage);

        //Object Detection on Live Webcam feed
        document.getElementById("Live-Webcam").onclick = function(){
            console.log("Object Detection on Live Webcam Feed");
            const video = document.getElementById("live-webcam-feed");
            const enableWebcam = document.getElementById("play-pause-webcam");
            let model = undefined;
            let streaming = false;
            let src;
            let cap;

            //Browser Feature Detection
            if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)){
                enableWebcam.addEventListener('click', ()=>{
                    if (!streaming){
                        console.log('Streaming Started');
                        enableCam();
                        streaming=true;
                    }
                    else{
                        console.log("Streaming Paused");
                        video.pause();
                        video.srcObject.getTracks().forEach(track => track.stop());
                        video.srcObject=null;
                        streaming=false;
                    }

                });
            }
            else{
                console.warn("getUseMedia() is not supported by your browser");
            }
        
            function enableCam(){
                if (!model){
                    return;
                }
                navigator.mediaDevices.getUserMedia({'video':true, 'audio':false}).then(function(stream){
                    video.srcObject=stream;
                    video.addEventListener('loadeddata', predictWebcam);

                });
            }
            setTimeout(async function(){
                try{
                    model = await tf.loadGraphModel("yolov8n_web_model/model.json")
                }
                catch(error){
                    console.log("Error Loading YOLOv8 tf.js model");
                }
            }, 0);

            async function predictWebcam(){
                //Check if the video element has loaded the data
                if (!video || !video.videoWidth || !video.videoHeight){
                    return;
                }
                console.log("Video Width", video.width, "Video Height", video.height);
                const begin = Date.now();
                src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
                cap = new cv.VideoCapture(video);
                cap.read(src);
                console.log("Width", src.cols, "Height", src.rows);
                const inputTensorShape = model.inputs[0].shape;
                const modelWidth = inputTensorShape[1];
                const modelHeight = inputTensorShape[2];
                console.log("Model Width", modelWidth, "Model height", modelHeight);
                const preprocess = (video, modelWidth, modelHeight) =>{
                    let xRatio,  yRatio;
                    const input = tf.tidy(()=>{
                        //Convert the Pixel Data From  Source into TensorFLow js tensor
                        const webcamFeed = tf.browser.fromPixels(video);
                        //Extracting the Width and Height of  Tensor
                        const [h,w] = webcamFeed.shape.slice(0,2);
                        //Height and Width of the  Tensor
                        console.log("Height", h, "Width", w);
                        //Max Value Between Width and Height
                        const maxSize = Math.max(w, h);
                        //Applying Padding
                        const webcamFeedPadded = webcamFeed.pad([
                            [0, maxSize - h],
                            [0, maxSize - w],
                            [0,0]
                        ]);
                        xRatio = maxSize/w;
                        yRatio = maxSize/h;
                        //Apply Bilinear Interpolation
                        return tf.image.resizeBilinear(webcamFeedPadded, [modelWidth, modelHeight]).div(255.0).expandDims(0);
                    })
                return [input, xRatio, yRatio]
                };
                const [input, xRatio, yRatio] = preprocess(video, modelWidth, modelHeight);
                console.log("Input Shape", input.shape, "X-Ratio", xRatio, "Y-Ratio", yRatio);
                //Pass Input  Tensor to the TensorFlow js Model
                const res = model.execute(input);
                //Rearrange the Dimensions of the Tensor
                //[batch_size, height, width] ==> [ batch_size, width, height];
                const transres = res.transpose([0,2,1]);
                const boxes = tf.tidy(()=>{
                    const w = transres.slice([0,0,2], [-1,-1,1]);
                    const h = transres.slice([0,0,3], [-1,-1,1]);
                    const x1 = tf.sub(transres.slice([0,0,0], [-1,-1,1]), tf.div(w,2));
                    const y1 = tf.sub(transres.slice([0,0,1], [-1,-1,1]), tf.div(h,2));
                    return tf.concat([y1, x1, tf.add(y1, h), tf.add(x1, w)], 2).squeeze();
                });
                //Calcualte the Confidence Score and Class Names
                const [scores, classes] = tf.tidy(() => {
                    const rawScores = transres.slice([0,0,4], [-1,-1, numClass]).squeeze(0);
                    return [rawScores.max(1), rawScores.argMax(1)];
                });
                //Applying Non Max Supression
                const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500,0.45, 0.60);
                const predictionsLength = nms.size;
                console.log("Predictions Length", predictionsLength)

                if(predictionsLength > 0){
                    const boxes_data = boxes.gather(nms, 0).dataSync();
                    const score_data = scores.gather(nms, 0).dataSync();
                    const classes_data = classes.gather(nms, 0).dataSync();
                    console.log("Boxes Data", boxes_data, "Score Data", score_data, "Classes Data", classes_data);
                    const xScale = src.cols/modelWidth;
                    const yScale = src.rows/ modelHeight;
                    console.log("Score Data Length", score_data.length);
                    for (let i=0; i < score_data.length; ++i){
                        const classID = classes_data[i];
                        const className = labels[classes_data[i]];
                        const ConfidenceScore = (score_data[i] * 100).toFixed(1);
                        let [y1, x1, y2, x2] = boxes_data.slice(i*4, (i+1)*4);
                        x1 *= xRatio * xScale;
                        x2 *= xRatio * xScale;
                        y1 *= yRatio * yScale;
                        y2 *= yRatio * yScale;
                        const height = y2 - y1;
                        const width = x2 - x1;
                        console.log(x1, y1, width, height, className, ConfidenceScore);
                        let point1 = new cv.Point(x1, y1);
                        let point2 = new cv.Point(x1+ width, y1 + height);
                        cv.rectangle(src, point1, point2, computeColorForLabels(classID), 4);
                        //const text = `${className} - ${Math.round(confidenceScore)/100}`
                        const text = className + " - " + ConfidenceScore + "%"
                        // Create a hidden canvas element to measure the text size
                        const canvas = document.createElement("canvas");
                        const context = canvas.getContext("2d");
                        context.font = "22px Arial"; // Set the font size and family as needed
                        // Measure the width of the text
                        const textWidth = context.measureText(text).width;
                        cv.rectangle(src, new cv.Point(x1,y1-20), new cv.Point(x1+ textWidth + context.lineWidth, y1), computeColorForLabels(classID),-1)
                        cv.putText(src, text, new cv.Point(x1, y1 - 5),cv.FONT_HERSHEY_TRIPLEX, 0.50, new cv.Scalar(255,255,255,255), 1);
                     }
                   cv.imshow("main-canvas", src);  
    
                }
            else{
                cv.imshow("main-canvas", src);
            }
            //Clear the Memory
            tf.dispose([res, transres, boxes, scores, classes, nms]);
            //Call the setTimeout function again after a delay
            const delay = 1000/FPS - (Date.now() - begin);
            setTimeout(predictWebcam, delay);
            //Release the Source Frame
            src.delete();
    
            }
        }
    }
}