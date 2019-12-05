import java.nio.file.Path;
import java.nio.file.Paths;

public class App {

    public static void main(String[] args) {

        Path currentPath = Paths.get("");
        String wav_folder = Paths.get(
                currentPath.toAbsolutePath().getParent().getParent().toString(),
                "dataset_wav")
                .toString();
        System.out.println(wav_folder);


    }

}
