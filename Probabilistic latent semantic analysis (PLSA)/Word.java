/**
 * Created by akash on 3/4/16.
 */
import java.util.Comparator;

public class Word implements Comparator<Word> ,Comparable<Word>{
    String word;
    double prob;

    public Word() {

    }

    public Word(String word, double prob) {
        this.word = word;
        this.prob = prob;
    }

    @Override
    public int compareTo(Word arg0) {
        //System.out.println(arg0);
        return Double.compare(arg0.prob, prob);
    }

    @Override
    public int compare(Word arg0, Word arg1) {
        return Double.compare(arg1.prob, arg0.prob);
    }
}
