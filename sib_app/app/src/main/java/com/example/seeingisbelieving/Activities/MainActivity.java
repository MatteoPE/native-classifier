package com.example.seeingisbelieving.Activities;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import com.example.seeingisbelieving.R;
import com.example.seeingisbelieving.Utils.FileUtils;
import com.example.seeingisbelieving.Utils.MFCC.MFCC;
import com.example.seeingisbelieving.Utils.Wav.WavFile;
import com.example.seeingisbelieving.Utils.Wav.WavFileException;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.io.File;
import java.io.IOException;


public class MainActivity extends AppCompatActivity {

    private static final int AUDIO_LOAD_CODE = 42;
    private static final String TAG_AUDIO_LOAD = "audio_load";
    public static final String MFCC_TAG = "mfccCreation";
    public static final String UPLOAD_TASK_TAG = "uploadTask";

    private static final int PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE_ = 1;
    private static final int PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE_ = 2;
    private static final int ALL_PERMISSIONS = 100;

    private boolean mStoragePermissionGranted;
    private boolean mReadStoragePermissionGranted;

    FirebaseStorage storage = FirebaseStorage.getInstance();
    // Create a storage reference from our app
    StorageReference storageRef = storage.getReference();
    // Create a child reference
    StorageReference mfccRef = storageRef.child("mfcc");
    private UploadTask uploadTask;

    private Button uploadAudio, uploadButton;

    private float[] mfccFeatureData;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        uploadAudio = (Button) findViewById(R.id.uploadAudio);
        uploadButton = (Button) findViewById(R.id.uploadButton);

        uploadAudio.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                performFileSearch();
            }
        });
        uploadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                uploadMfccFeatures();
            }
        });

        mStoragePermissionGranted = false;
        mReadStoragePermissionGranted = false;

        getPermissions();

    }

    private void uploadMfccFeatures() {
        // Set the model name and id of the phone
        String phone_id = Build.MODEL.replace(" ", "") + "_" + Settings.System.getString(getContentResolver(), Settings.Secure.ANDROID_ID);

        // Setup the directory name and create the directory
        String directory = Environment.getExternalStorageDirectory().getAbsolutePath() + "/mfcc/";
        FileUtils.createDirectory(directory);

        // Setup the file names for the recording files
        String txt_file = directory + "mfcc_file.txt";

        Uri txtFile = Uri.fromFile(new File(txt_file));

        StorageReference currRef = mfccRef.child(txtFile.getLastPathSegment());

        uploadTask = currRef.putFile(txtFile);

        // Register observers to listen for when the download is done or if it fails
        uploadTask.addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception exception) {
                // Handle unsuccessful uploads
                Log.i(UPLOAD_TASK_TAG,"Upload on Firebase Storage failed: " + exception.getMessage());
            }
        }).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
            @Override
            public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                // taskSnapshot.getMetadata() contains file metadata such as size, content-type, etc.
                Log.i(UPLOAD_TASK_TAG,"Upload on Firebase Storage succesfully done" );
            }
        });

    }

    /**
    * Fires an intent to spin up the "file chooser" UI and select an image.
    */
    public void performFileSearch() {

        // ACTION_OPEN_DOCUMENT is the intent to choose a file via the system's file
        // browser.
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);

        // Filter to only show results that can be "opened", such as a
        // file (as opposed to a list of contacts or timezones)
        intent.addCategory(Intent.CATEGORY_OPENABLE);

        // Filter to show only images, using the image MIME data type.
        // If one wanted to search for ogg vorbis files, the type would be "audio/ogg".
        // To search for all documents available via installed storage providers,
        // it would be "*/*".
        intent.setType("audio/*");

        startActivityForResult(intent, AUDIO_LOAD_CODE);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode,
                                 Intent resultData) {

        // The ACTION_OPEN_DOCUMENT intent was sent with the request code
        // AUDIO_LOAD_CODE. If the request code seen here doesn't match, it's the
        // response to some other intent, and the code below shouldn't run at all.

        if (requestCode == AUDIO_LOAD_CODE && resultCode == Activity.RESULT_OK && resultData != null) {
            // The document selected by the user won't be returned in the intent.
            // Instead, a URI to that document will be contained in the return intent
            // provided to this method as a parameter.
            // Pull that URI using resultData.getData().
            Uri uri = null;
            if (resultData != null) {
                uri = resultData.getData();
                Log.i(TAG_AUDIO_LOAD, "Uri: " + uri.toString());
                String path = getRealPathFromURI(uri);
                createMfcc(path);
                Log.i("path", path);
            }
        }
    }

    public void createMfcc(String strFilename){
        File file = new File(strFilename);

        WavFile wavFile = null;

        try {
            wavFile = WavFile.openWavFile(file);
        } catch (IOException e) {
            Log.i(MFCC_TAG, e.getMessage());
            return;
        } catch (WavFileException e) {
            Log.i(MFCC_TAG, e.getMessage());
            return;
        }

        int numChannels = wavFile.getNumChannels();

        long num_frames = wavFile.getNumFrames();

        double[] buffer = new double[(int)num_frames];

        int framesRead;

        try {
            framesRead = wavFile.readFrames(buffer, (int)num_frames);
        } catch (IOException e) {
            Log.i(MFCC_TAG, e.getMessage());
            return;
        } catch (WavFileException e) {
            Log.i(MFCC_TAG, e.getMessage());
            return;
        }

        MFCC Obj_Mfcc = new MFCC();
        float[] mfcc_feature =  Obj_Mfcc.process(buffer);

        Log.i(MFCC_TAG, mfcc_feature.toString());

        mfccFeatureData = mfcc_feature;

        uploadButton.setVisibility(View.VISIBLE);
        uploadButton.setClickable(true);

        return;

    }


    private void getPermissions() {
        final String[] permissions = new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE};
        if (
                (ContextCompat.checkSelfPermission(this.getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE)
                        == PackageManager.PERMISSION_GRANTED) &&
                (ContextCompat.checkSelfPermission(this.getApplicationContext(), Manifest.permission.READ_EXTERNAL_STORAGE)
                        == PackageManager.PERMISSION_GRANTED)
        ) {
            mStoragePermissionGranted = true;
            mReadStoragePermissionGranted = true;
        } else {
            ActivityCompat.requestPermissions(this,
                    permissions,
                    ALL_PERMISSIONS);
        }
    }


    private void getStoragePermission() {

        if (ContextCompat.checkSelfPermission(this.getApplicationContext(),
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED) {
            mStoragePermissionGranted = true;
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE_);
        }
    }

    private void getReadStoragePermission() {

        if (ContextCompat.checkSelfPermission(this.getApplicationContext(),
                Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED) {
            mReadStoragePermissionGranted = true;
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE_);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String permissions[],
                                           @NonNull int[] grantResults) {

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode) {
            case ALL_PERMISSIONS: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED
                        && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                }
                else {
                    getPermissions();
                }
            }

            case PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE_: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                }
                else {
                    getStoragePermission();
                }
            }

            case PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE_: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                }
                else {
                    getReadStoragePermission();
                }
            }
        }
    }

    private String getRealPathFromURI(Uri contentURI) {

        String result;
        String[] filePathColumn = { MediaStore.Images.Media.DATA };

        Cursor cursor = getContentResolver().query(contentURI, filePathColumn, null, null, null);
        if (cursor == null) { // Source is Dropbox or other similar local file path
            result = contentURI.getPath();
        } else {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(filePathColumn[0]);
            result = cursor.getString(idx);
            cursor.close();
        }
        return result;
    }

}
