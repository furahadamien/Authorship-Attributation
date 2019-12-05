package main;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;

public class Main {

    private static String BASE_URL = "https://www.fanfiction.net/book/Harry-Potter/";
    private static String PAGE_URL = BASE_URL + "?&srt=1&lan=1&r=10&len=5&p="; // filtered for English only, >5k words

    private static HashSet<String> visitedStories = new HashSet<>();

    public static void main(String[] args) {
        for (int i = 2; i < 12; i++) {
            System.out.println(i);
            try {
                Document doc = Jsoup.connect(PAGE_URL + i).get();
                Elements links = doc.select("a.stitle[href]");
                for (Element storyLink : links) {
                    String storyURL = storyLink.attr("href");
                    System.out.println(storyURL);

                    if (visitedStories.contains(storyURL)) {
                        continue;
                    } else {
                        visitedStories.add(storyURL);
                    }
                    // get the document representing the post
                    Document page = Jsoup.connect("https://www.fanfiction.net" + storyURL).get();
                    Elements authorTags = page.select("a.xcontrast_txt[href]");
                    String author = "no author found";
                    for (Element e : authorTags) {
                        if (e.attr("href").startsWith("/u/")) {
                            author = e.attr("href");
                            break;
                        }
                    }

                    Element chap_select = page.select("select#chap_select option").last();
                    int maxChap = 1;
                    if (chap_select != null)
                        maxChap = Integer.parseInt(chap_select.attr("value"));

                    StringBuilder body = new StringBuilder();
                    String[] fields = storyURL.split("/");

                    for (int nextChap = 1; nextChap <= maxChap; nextChap++) {
                        fields[3] = "" +nextChap;
                        String chapURL = String.join("/", fields);
                        Document chap = Jsoup.connect("https://www.fanfiction.net" + chapURL).get();
                        Element storyText = chap.select("div#storytext").first();
                        body.append(storyText.text());
                        body.append("\n");
                    }

                    String output = author.replace("/", "-") + ".txt";
                    BufferedWriter bw = new BufferedWriter(new FileWriter("files/" +output));
                    bw.write(body.toString());
                    bw.close();
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

}
