import * as tf from "@tensorflow/tfjs";
import { PGAgent } from "../agents";

/**
 * Conventional neuroevolution
 * 
 */

// 1 Make n agents
// 2 play a game with each agent and record the score
// 3 sort the agents by score and keep the top m agents
// 4 put the weights and biases of each agent into a chromosome
//     either a tf.TensorBuffer or Array
// 5 make n / m copies of each selected chromosome
// 6 perform crossover and mutation on each new chromosome (old chromosomes stay the same)
//     crossover - swap values in a range of chromosome indices that may or may not be contiguous
//     mutation - modify the value of a chromosome at a given index by some epsilon
// 7 make n - m new agents and initialize their weights and biases with the data
//    from the new chromosomes
// repeat 2 - 7 until your neurons have evolved
