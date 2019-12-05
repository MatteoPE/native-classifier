import Wav.WavFile;
import Wav.WavFileException;
import mfcc.MFCC;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException, WavFileException {

        Path currentPath = Paths.get("");
        String wav_folder = Paths.get(
                currentPath.toAbsolutePath().getParent().getParent().toString(),
                "dataset_wav")
                .toString();
        File directory = new File(wav_folder);

        FileFilter directoryFileFilter = new FileFilter() {
            public boolean accept(File file) {
                return file.isDirectory();
            }
        };

        File[] directoryListAsFile = directory.listFiles(directoryFileFilter);
        List<String> foldersInDirectory = new ArrayList<String>(directoryListAsFile.length);

        for (File directoryAsFile : directoryListAsFile) {

            String folder_path_wav = directoryAsFile.getPath();
            String folder_path_feature = folder_path_wav.replace("dataset_wav","dataset_java_feature");

            //make directory if not exists
            File directory_feature = new File(folder_path_feature);
            if (! directory_feature.exists()){
                directory_feature.mkdirs();
                // If you require it to make the entire directory path including parents,
                // use directory.mkdirs(); here instead.
            }


            File[] fileNames = directoryAsFile.listFiles();

            for(File file : fileNames) {

                //Get Mfcc Features
                float[] mfcc_feature = ExtractMfccFeature(file.getPath());

                String feature_file_path = folder_path_feature + "/" + file.getName().replace(".wav",".txt");

                //wite mfcc array to txt file
                File _file = new File(feature_file_path);
                BufferedWriter writer = new BufferedWriter(new FileWriter(_file.getAbsolutePath()));

                writer.write(
                        Arrays.toString(mfcc_feature)
                                .replace(",", "")
                                .replace("[", "")
                                .replace("]", ""));
                System.out.println(file.getPath() + " " + mfcc_feature.length);
                writer.close();

                }
            }

        }







    static float[] ExtractMfccFeature(String file_path) throws IOException, WavFileException {




        String strFilename  = file_path;


        // Open the wav file specified as the path
        File file = new File(strFilename);
        WavFile wavFile = WavFile.openWavFile(file);

        // Display information about the wav file
        //wavFile.display();

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

        return mfcc_feature;

    }
}
