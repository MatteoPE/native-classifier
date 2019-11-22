import Wav.WavFile;
import Wav.WavFileException;
import mfcc.MFCC;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) throws IOException, WavFileException {

        //System.out.println("Hello World!");



        myMethod();

    }

    static void myMethod() throws IOException, WavFileException {


        String strFilename  = "AAMNG1001.wav";


        // Open the wav file specified as the path
        File file = new File(strFilename);
        WavFile wavFile = WavFile.openWavFile(file);

        // Display information about the wav file
        wavFile.display();

        // Get the number of audio channels in the wav file
        int numChannels = wavFile.getNumChannels();

        long num_frames = wavFile.getNumFrames();
        // Create a buffer of 100 frames
        double[] buffer = new double[(int)num_frames];




        int framesRead;

        // Read frames into buffer
        framesRead = wavFile.readFrames(buffer, (int)num_frames);

        MFCC Obj_Mfcc = new MFCC();
        float[] mfcc_feature =  Obj_Mfcc.process(buffer);

        System.out.println(Arrays.toString(mfcc_feature));

    }
}
