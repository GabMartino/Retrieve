


 var canvas, ctx, flag = false, prevX = 0, currX = 0, prevY = 0, currY = 0, dot_flag = false;
var websocket;
var x = "black", y = 2;
function isCanvasBlank(canvas) {
    return !canvas.getContext('2d')
      .getImageData(0, 0, canvas.width, canvas.height).data
      .some(channel => channel !== 0);
  }

function init() {
    websocket = new WebSocket("ws://localhost:8082/ws");

    results = document.getElementById("Results")
    websocket.onmessage = function (obj) {
        json = JSON.parse(obj.data).data
        console.log(json)
        for (const o of json) {
            div = document.createElement("div")
            div.setAttribute("style", "margin: auto;")
            img = document.createElement("img")
            img.setAttribute("style", "height:200px; width:200px; margin: 2% 2% 2% 2%;")
            img.setAttribute("src", "data:image/png;base64,"+o["imageData"])
            p = document.createElement("p")
            console.log(o["className"])
            if (o["className"] == null){
                o["className"] = "Noise"
            }
            p.innerHTML = o["className"]
            div.appendChild(p)
            div.appendChild(img)

            results.appendChild(div)

        }
    }
    canvas = document.getElementById('canvasimg');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;
    console.log(w, h)
    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);



        const fileSelector = document.getElementById('Upload');
          var reader = new FileReader();
         fileSelector.addEventListener('change', (event) => {
         const fileList = event.target.files;
         reader.readAsDataURL(fileList[0])

      });
         reader.onload = function (event){
            sendObject(reader.result)
             results.innerHTML = ''
         }
}
function sendObject(image){

    data = {
        image: image
    }
    websocket.send(JSON.stringify(data))
}
function draw() {
  ctx.beginPath();
  ctx.moveTo(prevX, prevY);
  ctx.lineTo(currX, currY);
  ctx.strokeStyle = x;
  ctx.lineWidth = y;
  ctx.stroke();
  ctx.closePath();
}

function erase() {
    ctx.clearRect(0, 0, w, h);
}

function save() {
    if (!isCanvasBlank(canvas)){
        var dataURL = canvas.toDataURL();
          //document.getElementById("canvasimg").style.border = "2px solid";
        sendObject(dataURL)
        results.innerHTML = ''
    }
  
}

function findxy(res, e) {
 if (res == 'down') {
     prevX = currX;
     prevY = currY;
     currX = e.clientX - canvas.offsetLeft;
     currY = e.clientY - canvas.offsetTop;

     flag = true;
     dot_flag = true;
     if (dot_flag) {
         ctx.beginPath();
         ctx.fillStyle = x;
         ctx.fillRect(currX, currY, 2, 2);
         ctx.closePath();
         dot_flag = false;
     }
 }
 if (res == 'up' || res == "out") {
     flag = false;
 }
 if (res == 'move') {
     if (flag) {
         prevX = currX;
         prevY = currY;
         currX = e.clientX - canvas.offsetLeft;
         currY = e.clientY - canvas.offsetTop;
         draw();
     }
 }
}