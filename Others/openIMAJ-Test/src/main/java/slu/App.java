package slu;

import org.openimaj.audio.AudioPlayer;
import org.openimaj.audio.SampleChunk;
import org.openimaj.audio.features.MFCC;
import org.openimaj.audio.processor.FixedSizeSampleAudioProcessor;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.video.xuggle.XuggleAudio;
import org.openimaj.vis.audio.AudioWaveform;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;


/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) {

        XuggleAudio xa = new XuggleAudio(new File("AAMNG1001.wav"));

        FixedSizeSampleAudioProcessor fp = new FixedSizeSampleAudioProcessor(xa,400,160);
        MFCC mfcc = new MFCC(fp);

        //MFCC mfcc = new MFCC(xa);

        SampleChunk sc = null;

        List<DoubleFV> temp = new ArrayList<DoubleFV>();

        while( (sc = mfcc.nextSampleChunk()) != null )
        {
            DoubleFV mfccs = mfcc.extractFeature(sc);
            System.out.println(mfccs);

            temp.add(mfccs);
        }



        //write mfcc feature vector to a file
        File file = new File("mfcc.txt");
        FileWriter fw = null;
        try {
            fw = new FileWriter(file);
            for(DoubleFV vector: temp){
                fw.write(String.valueOf(vector)+System.lineSeparator());
            }


        }catch (IOException e){
            e.printStackTrace();
        } finally{
            try{
                fw.close();
            }catch(IOException e){
                e.printStackTrace();
            }
        }

        System.out.println("\nsuccess");
    }
}
