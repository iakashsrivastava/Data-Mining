import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.io.File;
import java.util.Collections;
import java.util.ArrayList;
import org.apache.commons.io.FileUtils;

public class main {

	public static void main(String[] args) throws NumberFormatException, IOException {
		// TODO Auto-generated method stub
		int numoftopics = 4;
		int DocSize = 20;
		String wordDictFile = "Data/term_dict.txt";
		String docTermFile = "Data/CT.txt";
		Plsa myPLSA = new Plsa(numoftopics);
		myPLSA.setDocSize(DocSize);
		myPLSA.readWordDict(wordDictFile);
		myPLSA.readDocTermMatrix(docTermFile);
		int maxIter = 1000;
		myPLSA.train(maxIter);	
		double[][] theta = myPLSA.getDocTopics();
		double[][] beta = myPLSA.getTopicWordPros();
		fileWrite("theta.txt", theta);
		fileWrite("beta.txt", beta);
		writeBeta( beta, myPLSA.getAllWords());
		writeTheta(theta);
	}

	public static void writeBeta(double[][] beta, List<String> wordsList)throws IOException {

		for (int index = 0; index < beta.length; index++) {

			List<Word> words = new ArrayList<>();

			for (int wordIndex = 0 ; wordIndex < wordsList.size(); wordIndex++)
				words.add(new Word( wordsList.get(wordIndex), beta[index][wordIndex]));


			Collections.sort(words, new Word());

			String top10Word ="";
			int counter =0;
			for (Word word : words) {
					if(++counter >= 11)
						break;
					top10Word += word.word + " Probability = " + word.prob + "\n";
			}
			fileWrite("betaTopic_"+index+".txt",top10Word);
		}
	}
	private static void writeTheta(double[][] theta) throws IOException {

		String topic = "";
		for (int index = 0; index < theta.length; index++) {

			List<Topic> topicList = new ArrayList<Topic>();

				for (int j = 0; j < theta[index].length; j++)
					topicList.add(new Topic(j + 1, theta[index][j]));

			Collections.sort(topicList, new Topic());
			topic += topicList.get(0).topic + " --> " + topicList.get(0).prob +"\n";
		}

		fileWrite("venueTopic.txt",topic);
	}

	public static void fileWrite(String fileName, String data)throws IOException {

		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("output/" + fileName)));
		bw.write(data);
		bw.close();
	}


	public static void fileWrite(String fileName, double[][] data)throws IOException {

		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("output/" + fileName)));

		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[0].length; j++)
				bw.write(data[i][j] + "\t");
			bw.write("\n");
		}
		bw.close();
	}
}
