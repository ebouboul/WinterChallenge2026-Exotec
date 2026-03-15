import com.codingame.gameengine.runner.MultiplayerGameRunner;
import com.codingame.gameengine.runner.simulate.GameResult;

public class Main {
    public static void main(String[] args) {
        if (args.length >= 2) {
            // HEADLESS MODE: Used by tuner.py
            MultiplayerGameRunner gameRunner = new MultiplayerGameRunner();
            
            // Apply the random seed injected by python
            if (args.length >= 3) {
                gameRunner.setSeed(Long.parseLong(args[2]));
            }
            
            gameRunner.addAgent(args[0], "Player 1");
            gameRunner.addAgent(args[1], "Player 2");

            // Run without the web server
            GameResult result = gameRunner.simulate();

            int score0 = result.scores.getOrDefault(0, -1);
            int score1 = result.scores.getOrDefault(1, -1);

            if (score0 > score1) {
                System.out.println("WINNER: 0");
            } else if (score1 > score0) {
                System.out.println("WINNER: 1");
            } else {
                System.out.println("WINNER: -1");
            }
        } else {
            // VISUAL MODE: Normal local testing in browser
            MultiplayerGameRunner gameRunner = new MultiplayerGameRunner();

            // Set seed here (leave commented for random)
            // gameRunner.setSeed(-1566415677164768800L);

            gameRunner.addAgent("python3 config/Boss.py", "Player 1");
            gameRunner.addAgent("python3 config/Boss.py", "Player 2");

            gameRunner.start(8888);
        }
    }
}