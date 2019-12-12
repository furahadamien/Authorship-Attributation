package main;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.process.*;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Demo {

    private static String ROOT_DIR = "C:\\Users\\irene.HADES\\Documents\\Comp 550\\ISG\\Code\\src\\resources\\";

    public static void main(String[] args) throws IOException {
        long startTime = System.currentTimeMillis();
        String source = args[0];
        String numAuthors = args[1];

        String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
        LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);

        Set<String> savedProfiles = new HashSet<>();
        getSavedProfiles(savedProfiles);

        List<String> knownAuthors = new ArrayList<>();
        List<HashMap<String, HashMap<String,Integer>>> knownProfiles = new ArrayList<>();

        try (Stream<Path> walk = Files.walk(Paths.get(ROOT_DIR, "known", source, numAuthors))) {
            // walk through the resource directory (known folder)
            List<String> kResult = walk.filter(Files::isRegularFile)
                    .map(x -> x.toString()).collect(Collectors.toList());

            // for each file in the directory, call demoDP
            kResult.forEach((file) -> {
                System.out.println("Computing profile for: " + file);
                String authorName = file.replace(ROOT_DIR + "known\\", "");
                System.out.println("authorName: " + authorName);

                knownAuthors.add(authorName);
                if (savedProfiles.contains(authorName)) {
                    HashMap<String, HashMap<String, Integer>> profile = new HashMap<>();
                    loadSavedProfiles(authorName, profile);
                    knownProfiles.add(profile);
                } else {
                    HashMap<String, HashMap<String, Integer>> profile = demoDP(lp, file);
                    knownProfiles.add(profile);
                    // After you've computed an author's profile, SAVE it!
                    saveProfile(authorName, profile);
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }

        // for each file in the directory, call demoDP
        try (Stream<Path> walk = Files.walk(Paths.get(ROOT_DIR, "unknown", source, numAuthors))) {
            // walk through the resource directory (known folder)
            List<String> ukResult = walk.filter(Files::isRegularFile)
                    .map(x -> x.toString()).collect(Collectors.toList());

            List<String> goldStandards = new ArrayList<>();
            List<String> predictedResults = new ArrayList<>();
            List<String> predictedScores = new ArrayList<>();
            List<String> margins = new ArrayList<>();
            List<String> meanScores = new ArrayList<>();

            // for each file in the directory, call demoDP
            ukResult.forEach((file) -> {
                if (file.contains("labels.txt")) {
                    try {
                        BufferedReader br = new BufferedReader(new FileReader(file));
                        String s = br.readLine();
                        while (s != null) {
                            goldStandards.add(s);
                            s = br.readLine();
                        }
                        br.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                else {
                    HashMap<String, HashMap<String, Integer>> unknown = demoDP(lp, file);
                    int bestAuthor = -1;
                    float bestScore = -1;
                    float nextBest = bestScore;
                    float mean = 0;
                    float[] stdevScores = new float[knownProfiles.size()];

                    String allScores = "";
                    for (int i = 0; i < knownProfiles.size(); i++) {
                        HashMap<String, HashMap<String, Integer>> profile = knownProfiles.get(i);
                        System.out.println("Comparing " + file.replace(ROOT_DIR, "") + " with profile: " + knownAuthors.get(i));
                        float score = Similarity(unknown, profile);
                        allScores += String.format("%.2f", score) + " ";
                        if (score > bestScore) {
                            nextBest = bestScore;
                            bestScore = score;
                            bestAuthor = i;
                        }
                        mean += score;
                        stdevScores[i] = score;
                    }

                    mean = mean / knownProfiles.size();
                    float stdev = 0;
                    for (int i = 0; i < knownProfiles.size(); i++) {
                        stdev += (stdevScores[i] - mean) * (stdevScores[i] - mean);
                    }
                    stdev = stdev / knownProfiles.size();
                    meanScores.add("(" + String.format("%.2f", mean) + ", " + String.format("%.2f", stdev) + ")");
                    margins.add("" + (bestScore - nextBest));
                    predictedScores.add(allScores);
                    predictedResults.add(knownAuthors.get(bestAuthor));
                    System.out.println("Author with highest similarity: " + knownAuthors.get(bestAuthor) + "\n");
                }
            });

            float sum = 0;
            try {
                BufferedWriter bw = new BufferedWriter(new FileWriter(ROOT_DIR + "results\\" + source + "_" + numAuthors + "_results.txt"));
                for (int i = 0; i < predictedResults.size(); i++) {
                    System.out.println("predicted: " + predictedResults.get(i));
                    System.out.println("nextBest: " + margins.get(i));
                    System.out.println("actual: " + goldStandards.get(i));

                    bw.write("real author: " + goldStandards.get(i) + "\tpredicted scores: ");
                    bw.write(meanScores.get(i) + "\t");
                    bw.write(predictedScores.get(i));
                    if (predictedResults.get(i).contains(goldStandards.get(i))) {
                        sum++;
                        bw.write("\n");
                    } else {
                        bw.write("\tdiff: " + margins.get(i) + "\n");
                    }
                }
                bw.write("\n\naccuracy: ");
                bw.write("" + (sum / predictedResults.size()));
                bw.write("\n");
                long totalTime = (System.currentTimeMillis() - startTime);
                bw.write(totalTime + "ms runtime");
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void getSavedProfiles(Set<String> savedProfiles) {
        try (Stream<Path> walk = Files.walk(Paths.get(ROOT_DIR, "saved"))) {
            // walk through the resource directory (known folder)
            List<String> result = walk.filter(Files::isRegularFile)
                    .map(x -> x.toString()).collect(Collectors.toList());

            // for each file in the directory, call demoDP
            result.forEach((file) -> {
                System.out.println("saved: " + file.replace(ROOT_DIR + "saved\\", ""));
                savedProfiles.add(file.replace(ROOT_DIR + "saved\\", ""));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void loadSavedProfiles(String savename, HashMap<String, HashMap<String, Integer>> emptyProfile) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(ROOT_DIR + "saved\\" + savename));
            String s = br.readLine();
            while (s != null) {
                String[] arr = s.split("\t");
                HashMap<String, Integer> featureVect = new HashMap<>();

                String[] arr2 = arr[1].split(" ");
                for (String pair : arr2) {
                    String[] keyVal = pair.split(",");
                    int value = Integer.parseInt(keyVal[1].substring(0, keyVal[1].length()-1));
                    featureVect.put(keyVal[0].substring(1), value);
                }
                emptyProfile.put(arr[0], featureVect);
                s = br.readLine();
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveProfile(String name, HashMap<String, HashMap<String, Integer>> profile) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(ROOT_DIR + "saved\\" + name));
            StringBuilder rep = new StringBuilder("");
            for (String key: profile.keySet()) {
                rep.append(key);
                rep.append("\t");
                HashMap<String, Integer> vector = profile.get(key);
                for (String f : vector.keySet()) {
                    String pair = "(" + f + "," + vector.get(f) + ")";
                    rep.append(pair);
                    rep.append(" ");
                }
                rep.append("\n");
            }
            bw.write(rep.toString());
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * demoDP demonstrates turning a file into tokens and then parse
     * trees.  Note that the trees are printed by calling pennPrint on
     * the Tree object.  It is also possible to pass a PrintWriter to
     * pennPrint if you want to capture the output.
     * This code will work with any supported language.
     */
    public static HashMap<String, HashMap<String, Integer>> demoDP(LexicalizedParser lp, String filename) {

        // This option shows loading, sentence-segmenting and tokenizing
        // a file using DocumentPreprocessor.
        TreebankLanguagePack tlp = lp.treebankLanguagePack(); // a PennTreebankLanguagePack for English
        GrammaticalStructureFactory gsf = null;
        if (tlp.supportsGrammaticalStructures()) {
            gsf = tlp.grammaticalStructureFactory();
        }
        // You could also create a tokenizer here (as below) and pass it
        // to DocumentPreprocessor

        ISG graph = new ISG();
        for (List<HasWord> sentence : new DocumentPreprocessor(filename)) {
            Tree parse = lp.apply(sentence);

            if (gsf != null) {
                GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
                // Collection tdl = gs.typedDependenciesCCprocessed();

                // Build up the ISG for the input document by adding the parse of the current sentence to the existing graph
                // Note that each sentence will begin with the ROOT-0 node, and identical nodes across sentences get automatically collapsed
                for (TypedDependency td : gs.typedDependenciesCCprocessed()) {
                    // prettyPrintTD(td);

                    // NB: this adds the two endpoints of the edge to the nodes list in graph in addition to adding the edge itself
                    graph.addEdge(td.gov().toString(), td.dep().toString(), td.reln().toString());
                }
            }
        }

        // Compute the feature matrix for the ISG using the shortest path
        HashMap<String, HashMap<String, Integer>> matrix = graph.extractFeatureMatrix();
        List<String> allFeatures = graph.getFeatures();
        return matrix;
    }

    // Takes in unknown document feature matrix D1 and known author feature matrix D2
    // Computes the cosine similarity score between them, using following formula:
    //      Similarity(D1, D2) = SUM_[i=1 to m] ( SUM_[j=1 to |V|] (f_D1[i][j] * f_D2[i][j] ) / (sqrt(SUM_[j=1 to |V|] (f_D1[i][j])^2) * sqrt(SUM_[j=1 to |V|] (f_D2[i][j])^2))
    public static float Similarity(HashMap<String, HashMap<String, Integer>> D1, HashMap<String, HashMap<String, Integer>> D2) {
        float score = 0;
        Set<String> features = new HashSet<>();
        // Loop through each endpoint in D1
        for (String endpoint : D1.keySet()) {
            // Skip every iteration where there is no path to the endpoint in the known profile document
            if (!D2.containsKey(endpoint)) {
                continue;
            }

            features = D1.get(endpoint).keySet().stream()
                        .filter(D2.get(endpoint).keySet()::contains)
                        .collect(Collectors.toSet());

            int numerator = 0;
            int denominatorA = 0;
            int denominatorB = 0;

            // Loop through each component in the feature vector for node 'endpoint'
            for (String f : features) {
                int val1 = D1.get(endpoint).get(f);
                int val2 = D2.get(endpoint).get(f);
                numerator += val1 * val2;
                denominatorA += (val1 * val1);
                denominatorB += (val2 * val2);
            }

            if (denominatorA == 0 && denominatorB == 0) {
                continue;
            }
            score += numerator / (Math.sqrt(denominatorA) * Math.sqrt(denominatorB));
        }

        // System.out.println("final score: " + score);
        return score;
    }

    /**
     * demoAPI demonstrates other ways of calling the parser with
     * already tokenized text, or in some cases, raw text that needs to
     * be tokenized as a single sentence.  Output is handled with a
     * TreePrint object.  Note that the options used when creating the
     * TreePrint can determine what results to print out.  Once again,
     * one can capture the output by passing a PrintWriter to
     * TreePrint.printTree. This code is for English.
     */
    public static void demoAPI(LexicalizedParser lp) {
        // This option shows parsing a list of correctly tokenized words
        String[] sent = { "This", "is", "an", "easy", "sentence", "." };
        List<CoreLabel> rawWords = SentenceUtils.toCoreLabelList(sent);
        Tree parse = lp.apply(rawWords);
        parse.pennPrint();
        System.out.println();

        // This option shows loading and using an explicit tokenizer
        String sent2 = "This is another sentence.";
        TokenizerFactory<CoreLabel> tokenizerFactory =
                PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
        Tokenizer<CoreLabel> tok =
                tokenizerFactory.getTokenizer(new StringReader(sent2));
        List<CoreLabel> rawWords2 = tok.tokenize();
        parse = lp.apply(rawWords2);

        TreebankLanguagePack tlp = lp.treebankLanguagePack(); // PennTreebankLanguagePack for English
        GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
        GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
        List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
        System.out.println(tdl);
        System.out.println();

        // You can also use a TreePrint object to print trees and dependencies
        TreePrint tp = new TreePrint("penn,typedDependenciesCollapsed");
        tp.printTree(parse);
    }
    public static void corePipeline(String[] args) throws IOException {
        // set up optional output files
        PrintWriter out;
        if (args.length > 1) {
            out = new PrintWriter(args[1]);
        } else {
            out = new PrintWriter(System.out);
        }
        PrintWriter xmlOut = null;
        if (args.length > 2) {
            xmlOut = new PrintWriter(args[2]);
        }

        // Create a CoreNLP pipeline. To build the default pipeline, you can just use:
        //   StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        // Here's a more complex setup example:
        //   Properties props = new Properties();
        //   props.put("annotators", "tokenize, ssplit, pos, lemma, ner, depparse");
        //   props.put("ner.model", "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
        //   props.put("ner.applyNumericClassifiers", "false");
        //   StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Add in sentiment
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, parse");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Initialize an Annotation with some text to be annotated. The text is the argument to the constructor.
        Annotation annotation;
        if (args.length > 0) {
            annotation = new Annotation(IOUtils.slurpFileNoExceptions(args[0]));
        } else {
            annotation = new Annotation("Kosgi Santosh sent an email to Stanford University. He didn't get a reply.");
        }

        // run all the selected Annotators on this text
        pipeline.annotate(annotation);

        // this prints out the results of sentence analysis to file(s) in good formats
        pipeline.prettyPrint(annotation, out);
        if (xmlOut != null) {
            pipeline.xmlPrint(annotation, xmlOut);
        }

        // Access the Annotation in code
        // The toString() method on an Annotation just prints the text of the Annotation
        // But you can see what is in it with other methods like toShorterString()
        out.println();
        out.println("The top level annotation");
        out.println(annotation.toShorterString());
        out.println();

        // An Annotation is a Map with Class keys for the linguistic analysis types.
        // You can get and use the various analyses individually.
        // For instance, this gets the parse tree of the first sentence in the text.
        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        if (sentences != null && ! sentences.isEmpty()) {
            CoreMap sentence = sentences.get(0);
            out.println("The keys of the first sentence's CoreMap are:");
            out.println(sentence.keySet());
            out.println();
            out.println("The first sentence is:");
            out.println(sentence.toShorterString());
            out.println();
            out.println("The first sentence tokens are:");
            for (CoreMap token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                out.println(token.toShorterString());
            }
            Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
            out.println();
            out.println("The first sentence parse tree is:");
            tree.pennPrint(out);
            out.println();
            out.println("The first sentence basic dependencies are:");
            out.println(sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class).toString(SemanticGraph.OutputFormat.LIST));
            out.println("The first sentence collapsed, CC-processed dependencies are:");
            SemanticGraph graph = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
            out.println(graph.toString(SemanticGraph.OutputFormat.LIST));

            // Print out dependency structure around one word
            // This give some idea of how to navigate the dependency structure in a SemanticGraph
            IndexedWord node = graph.getNodeByIndexSafe(5);
            // The below way also works
            // IndexedWord node = new IndexedWord(sentences.get(0).get(CoreAnnotations.TokensAnnotation.class).get(5 - 1));
            out.println("Printing dependencies around \"" + node.word() + "\" index " + node.index());
            List<SemanticGraphEdge> edgeList = graph.getIncomingEdgesSorted(node);
            if (! edgeList.isEmpty()) {
                assert edgeList.size() == 1;
                int head = edgeList.get(0).getGovernor().index();
                String headWord = edgeList.get(0).getGovernor().word();
                String deprel = edgeList.get(0).getRelation().toString();
                out.println("Parent is word \"" + headWord + "\" index " + head + " via " + deprel);
            } else  {
                out.println("Parent is ROOT via root");
            }
            edgeList = graph.outgoingEdgeList(node);
            for (SemanticGraphEdge edge : edgeList) {
                String depWord = edge.getDependent().word();
                int depIdx = edgeList.get(0).getDependent().index();
                String deprel = edge.getRelation().toString();
                out.println("Child is \"" + depWord + "\" (" + depIdx + ") via " + deprel);
            }
            out.println();


            // Access coreference. In the coreference link graph,
            // each chain stores a set of mentions that co-refer with each other,
            // along with a method for getting the most representative mention.
            // Both sentence and token offsets start at 1!
            out.println("Coreference information");
            Map<Integer, CorefChain> corefChains =
                    annotation.get(CorefCoreAnnotations.CorefChainAnnotation.class);
            if (corefChains == null) { return; }
            for (Map.Entry<Integer,CorefChain> entry: corefChains.entrySet()) {
                out.println("Chain " + entry.getKey());
                for (CorefChain.CorefMention m : entry.getValue().getMentionsInTextualOrder()) {
                    // We need to subtract one since the indices count from 1 but the Lists start from 0
                    List<CoreLabel> tokens = sentences.get(m.sentNum - 1).get(CoreAnnotations.TokensAnnotation.class);
                    // We subtract two for end: one for 0-based indexing, and one because we want last token of mention not one following.
                    out.println("  " + m + ", i.e., 0-based character offsets [" + tokens.get(m.startIndex - 1).beginPosition() +
                            ", " + tokens.get(m.endIndex - 2).endPosition() + ')');
                }
            }
            out.println();

            out.println("The first sentence overall sentiment rating is " + sentence.get(SentimentCoreAnnotations.SentimentClass.class));
        }
        IOUtils.closeIgnoringExceptions(out);
        IOUtils.closeIgnoringExceptions(xmlOut);
    }

    private static void prettyPrintTD(TypedDependency td) {
        System.out.print(td + "\t\t --> \t\t");
        System.out.print(td.dep() + "\t");
        System.out.print(td.reln() + "\t");
        System.out.print(td.gov() + "\t");
        System.out.println();
    }

}