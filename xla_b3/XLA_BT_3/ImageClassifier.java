import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk; // KNN
import weka.classifiers.trees.J48; // Decision Tree
import weka.classifiers.functions.SMO; // SVM
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ImageClassifier {

    // Hàm huấn luyện mô hình
    public static Classifier trainModel(String algorithm, Instances trainingData) throws Exception {
        Classifier classifier;
        switch (algorithm) {
            case "KNN":
                classifier = new IBk(); // KNN
                break;
            case "SVM":
                classifier = new SMO(); // SVM
                break;
            case "DecisionTree":
                classifier = new J48(); // Decision Tree
                break;
            default:
                throw new IllegalArgumentException("Unknown algorithm");
        }
        classifier.buildClassifier(trainingData);
        return classifier;
    }

    // Hàm đánh giá mô hình
    public static void evaluateModel(Classifier classifier, Instances testData) throws Exception {
        // Đo thời gian và các độ đo như accuracy, precision, recall
        long startTime = System.nanoTime();
        Evaluation eval = new Evaluation(testData);
        eval.evaluateModel(classifier, testData);
        long endTime = System.nanoTime();
        System.out.println("Time: " + (endTime - startTime) / 1e6 + " ms");
        System.out.println("Accuracy: " + eval.pctCorrect());
        System.out.println("Precision: " + eval.precision(1));
        System.out.println("Recall: " + eval.recall(1));
    }

    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("path/to/dataset.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Chia dữ liệu thành train/test
        Instances train = new Instances(data, 0, (int) (data.numInstances() * 0.8));
        Instances test = new Instances(data, (int) (data.numInstances() * 0.8), data.numInstances());

        // Huấn luyện mô hình KNN
        Classifier knnModel = trainModel("KNN", train);
        evaluateModel(knnModel, test);

        // Huấn luyện mô hình SVM
        Classifier svmModel = trainModel("SVM", train);
        evaluateModel(svmModel, test);

        // Huấn luyện mô hình Decision Tree
        Classifier dtModel = trainModel("DecisionTree", train);
        evaluateModel(dtModel, test);
    }
}
