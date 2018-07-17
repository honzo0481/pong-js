import * as tf from "@tensorflow/tfjs";
import { Game } from "..";

/**
 * The  model is a simple two layer neural net with 1 hidden layer
 * of 25 relu nodes and a sigmoid output layer.
 * TODO: talk about loss function and optimizer
 * TODO: hyperparameters
 */
export class GeneticAgent {
  game: Game;
  player: string;
  actions: any = {
    'player1': [
      // up
      () => {
        this.game.setP1InputUp(true);
        this.game.setP1InputDown(false);
      },
      // down
      () => {
        this.game.setP1InputUp(false);
        this.game.setP1InputDown(true);
      }
    ],
    'player2': [
      // up
      () => {
        this.game.setP2InputUp(true);
        this.game.setP2InputDown(false);
      },
      // down
      () => {
        this.game.setP2InputUp(false);
        this.game.setP2InputDown(true);
      }
    ]
  }

  /** size of the model's hidden layer */
  hiddenUnits = 25;

  /** the neural net - outputs the probability of an action given the current state */
  model = tf.sequential();
  /** hidden layer of the neural net */
  //  TODOs:
  //    hidden.config.inputShape = State.shape
  //    State.shape = paddleLen + paddlePos*2 + ballPos(x, y) + ballV, + ballRad
  hidden = tf.layers.dense({
    units: this.hiddenUnits, 
    activation: 'relu', 
    inputShape: [8],
    kernelInitializer: 'glorotUniform',
    name: 'hidden'
  });
  /** outputs the highest probability action given the observed state */
  //  switch activation to softmax to handle num actions > 2
  output = tf.layers.dense({units: 1, activation: 'sigmoid'});

  constructor(game: Game, player: string) {
    this.game = game;
    this.player = player;
    // build the model 
    this.model.add(this.hidden);
    this.model.add(this.output);
  }

  /** Stochastically determine the next action for the given state according to the policy */
  nextAction(state: Array<number>) {
    // set the output layer's activation to softmax and use tf.argmax() to handle more than 2 actions
    const prediction = tf.tidy(() => {
      const stateTensor = tf.tensor2d(state, [1, 8]);
      return this.model.predict(stateTensor, {batchSize: 1});
    }) as tf.Tensor;
    const action = tf.tidy(() => {
      const output = prediction.squeeze().dataSync()[0];
      return output;
    })
    const choice = Math.random() < action ? 1 : 0;
    this.actions[this.player][choice]();
    prediction.dispose();
  }
}
