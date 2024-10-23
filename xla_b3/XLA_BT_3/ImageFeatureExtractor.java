import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class ImageFeatureExtractor {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static Mat loadImage(String filePath) {
        return Imgcodecs.imread(filePath);
    }

    // Hàm chuyển đổi ảnh thành vector đặc trưng
    public static double[] extractFeatures(Mat image) {
        Mat grayImage = new Mat();
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);
        return grayImage.get(0, 0);
    }
}
