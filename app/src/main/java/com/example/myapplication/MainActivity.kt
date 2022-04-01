package com.example.myapplication

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.*
import com.example.myapplication.databinding.ActivityMainBinding
import com.example.myapplication.ml.MobilenetQuant
import com.google.firebase.storage.FirebaseStorage
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File

lateinit var binding: ActivityMainBinding
class MainActivity : AppCompatActivity() {

    lateinit var select_image_button : ImageButton
    lateinit var make_prediction : ImageButton
    lateinit var img_view : ImageView
    lateinit var img_view2 : ImageView
    lateinit var text_view : TextView
    lateinit var text_view2 : TextView
    lateinit var bitmap: Bitmap
    lateinit var camerabtn : ImageButton

    public fun checkandGetpermissions(){
        if(checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
        }
        else{
            Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(requestCode == 100){
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
            }
            else{
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

            select_image_button = findViewById(R.id.button)
            make_prediction = findViewById(R.id.button2)
            img_view = findViewById(R.id.imageView2)
            text_view = findViewById(R.id.textView)
            camerabtn = findViewById(R.id.camerabtn)
            img_view2 = findViewById(R.id.imageview)
            // handling permissions
            checkandGetpermissions()

        val labels = application.assets.open("labels.txt").bufferedReader().use { it.readText() }.split("\n")
        val label = application.assets.open("label.txt").bufferedReader().use { it.readText() }.split("\n")



        select_image_button.setOnClickListener(View.OnClickListener {
            Log.d("mssg", "button pressed")
            var intent : Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"

            startActivityForResult(intent, 250)
        })



        //fetch----------

//        val imagename = findViewById<TextView>(R.id.we)as TextView

//        binding = ActivityMainBinding.inflate(layoutInflater)
//        setContentView(binding.root)

//        binding.getImage.setOnClickListener{
//            val imagename = binding.etImageid.text.toString()
//
//            val storageRef = FirebaseStorage.getInstance().reference.child("images/$imagename.jpg")
//
//            val localfile = File.createTempFile("tempImage","jpg")
//            storageRef.getFile(localfile).addOnSuccessListener {
//
//
//
//                val bitmap = BitmapFactory.decodeFile(localfile.absolutePath)
//                binding.imageview.setImageBitmap(bitmap)
//
//            }.addOnFailureListener{
//
//
//
//                Toast.makeText(this,"Failed to retrieve the image",Toast.LENGTH_SHORT).show()
//            }
//
//        }

        //fetch----------------------------------



        make_prediction.setOnClickListener(View.OnClickListener {

            var resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            val model = MobilenetQuant.newInstance(this)

            var tbuffer = TensorImage.fromBitmap(resized)
            var byteBuffer = tbuffer.buffer

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
            inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)

            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            var max = getMax(outputFeature0.floatArray)

            text_view.setText(labels[max]).toString()


            val imagename = (label[max])


            //Toast.makeText(this,imagename,Toast.LENGTH_SHORT).show()
            Toast.makeText(this,"Showing result...",Toast.LENGTH_SHORT).show()

            val storageRef = FirebaseStorage.getInstance().reference.child("images/$imagename.png")

            val localfile = File.createTempFile("tempImage","jpg")
            storageRef.getFile(localfile).addOnSuccessListener {

                val bitmap = BitmapFactory.decodeFile(localfile.absolutePath)
                img_view2.setImageBitmap(bitmap)

            }.addOnFailureListener{

                Toast.makeText(this,"No Result Found",Toast.LENGTH_SHORT).show()
            }

// Releases model resources if no longer used.
            model.close()
        })

        camerabtn.setOnClickListener(View.OnClickListener {
            var camera : Intent = Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(camera, 200)
        })


    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(requestCode == 250){
            img_view.setImageURI(data?.data)

            var uri : Uri?= data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
        }
        else if(requestCode == 200 && resultCode == Activity.RESULT_OK){
            bitmap = data?.extras?.get("data") as Bitmap
            img_view.setImageBitmap(bitmap)
        }
    }

    fun getMax(arr:FloatArray) : Int{
        var ind = 0;
        var min = 0.0f;

        for(i in 0..1000)
        {
            if(arr[i] > min)
            {
                min = arr[i]
                ind = i;
            }
        }
        return ind
    }
}

