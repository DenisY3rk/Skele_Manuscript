var net;
var count = 0;

const imgE1 = document.getElementById("img");
const descE1 = document.getElementById("descripcion-imagen");

const webcamE1 = document.getElementById("webcam");
const descCam1 = document.getElementById("console");

const clasificador = knnClassifier.create();
var webcam;

async function app (){
//descargar el modelo pre-entrenado
net = await mobilenet.load();
webcam = await tf.data.webcam (webcamE1);
    while(true){
        const img = await webcam.capture(); 
        const result = await net.classify (img);

        const activation = net.infer(img,"conv_preds");
        var result2;
        try{
            result2 = await clasificador.predictClass(activation);
        }catch(error){
            result2={}
        }
        const clases=["No Entrenado", "Denis", "tijeras", "fosforo", "alcohol", "celular"];
        

            

            try{
                document.getElementById("console2").innerHTML= "ORACION:" + clases[result2.label];
            }catch (error){
            document.getElementById("console2").innerHTML="No entrenado";
            }
        

        //eliminar el tensor de la memoria
        img.dispose();
        await tf.nextFrame;
    }
}
    imgE1.onload = async function(){
        displayImagePrediction();
    }

    async function displayImagePrediction(){
        try {
            const result = await net.classify(imgE1);
            descE1.innerHTML = JSON.stringify(result);
        } catch (error) {

        }
    }
    //cambiar imagen
    async function cambiarImagen(){
    }
    //añadir ejemplos al clasificador kNN
    async function addExample(classId) {
        console.log("Ejemplo agregado");
        const img = await webcam.capture();

        //transferir know
        const activation = net.infer(img,true);
        
        //añadir la clase
        clasificador.addExample(activation,classId);
        img.dispose();
        //MOSTRAR ORACION
        
    }

app();