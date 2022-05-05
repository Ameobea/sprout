import * as tf from '@tensorflow/tfjs';
import { type LayerVariable, type Shape, type Tensor, tidy } from '@tensorflow/tfjs';
import { RNNCell, StackedRNNCells, type RNNLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/recurrent';
import * as K from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';
import { getInitializer, Initializer, type InitializerIdentifier } from '@tensorflow/tfjs-layers/dist/initializers';
import { getExactlyOneShape, isArrayOfShapes } from '@tensorflow/tfjs-layers/dist/utils/types_utils';
import { Activation, getActivation } from '@tensorflow/tfjs-layers/dist/activations';
import { assertPositiveInteger } from '@tensorflow/tfjs-layers/dist/utils/generic_utils';
import type { Kwargs } from '@tensorflow/tfjs-layers/dist/types';
import type { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import type { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { nameScope } from '@tensorflow/tfjs-layers/dist/common';

export class MyStackedRNNCells extends StackedRNNCells {
  public cells: MySimpleRNNCell[];

  public build(inputShape: Shape | Shape[]): void {
    if (isArrayOfShapes(inputShape)) {
      inputShape = (inputShape as Shape[])[0];
    }
    inputShape = inputShape as Shape;
    let outputDim: number;
    this.cells.forEach((cell, i) => {
      nameScope(`RNNCell_${i}`, () => {
        cell.build(inputShape);
        outputDim = cell.outputDim;

        inputShape = [inputShape[0], outputDim] as Shape;
      });
    });
    this.built = true;
  }
}

interface MySimpleRNNCellLayerArgs extends LayerArgs {
  /**
   * stateSize: Positive integer, size of state vector passed between
   * steps of the sequence.
   */
  stateSize: number;
  /**
   * outputSize: Positive integer, dimensionality of the output space.
   */
  outputDim: number;
  /**
   * Activation function to use for the recurrent tree.
   * Default: hyperbolic tangent ('tanh').
   * If you pass `null`,  'linear' activation will be applied.
   */
  recurrentActivation?: ActivationIdentifier;
  /**
   * Activation function to use for the output tree.
   * Default: hyperbolic tangent ('tanh').
   * If you pass `null`,  'linear' activation will be applied.
   */
  outputActivation?: ActivationIdentifier;
  /**
   * Whether the recursive tree uses a bias vector.
   */
  useRecurrentBias?: boolean;
  /**
   * Whether the output tree uses a bias vector.
   */
  useOutputBias?: boolean;
  /**
   * Initializer for the `kernel` weights matrix, used for the linear
   * transformation of the inputs.
   */
  kernelInitializer?: InitializerIdentifier | Initializer;
  /**
   * Initializer for the `recurrentKernel` weights matrix, used for
   * linear transformation of the recurrent state.
   */
  recurrentInitializer?: InitializerIdentifier | Initializer;
  /**
   * Initializer for the bias vector.
   */
  biasInitializer?: InitializerIdentifier | Initializer;
}

export class MySimpleRNNCell extends RNNCell {
  /** @nocollapse */
  static className = 'SimpleRNNCell';
  readonly stateSize: number;
  readonly outputDim: number;
  readonly recurrentActivation: Activation;
  readonly outputActivation: Activation;
  readonly useOutputBias: boolean;
  readonly useRecurrentBias: boolean;

  readonly kernelInitializer: Initializer;
  readonly recurrentInitializer: Initializer;
  readonly biasInitializer: Initializer;

  outputTree: LayerVariable;
  outputBias: LayerVariable;
  recurrentTree: LayerVariable;
  recurrentBias: LayerVariable;

  readonly DEFAULT_ACTIVATION = 'tanh';
  readonly DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
  readonly DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
  readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier = 'zeros';

  constructor(args: MySimpleRNNCellLayerArgs) {
    super(args);
    this.outputDim = args.outputDim;
    assertPositiveInteger(this.outputDim, 'outputSize');
    this.recurrentActivation = getActivation(
      args.recurrentActivation == null ? this.DEFAULT_ACTIVATION : args.recurrentActivation
    );
    this.outputActivation = getActivation(
      args.outputActivation == null ? this.DEFAULT_ACTIVATION : args.outputActivation
    );
    this.useRecurrentBias = args.useRecurrentBias == null ? true : args.useRecurrentBias;
    this.useOutputBias = args.useOutputBias == null ? true : args.useOutputBias;

    this.kernelInitializer = getInitializer(args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.recurrentInitializer = getInitializer(args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);

    this.biasInitializer = getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);

    this.stateSize = args.stateSize;
    // assertPositiveInteger(this.stateSize, `stateSize`);
  }

  build(inputShape: Shape | Shape[]): void {
    inputShape = getExactlyOneShape(inputShape) as Shape;
    this.outputTree = this.addWeight(
      'output_tree',
      [inputShape[inputShape.length - 1] + this.stateSize, this.outputDim],
      null,
      this.kernelInitializer,
      undefined,
      true
    );
    if (this.useOutputBias) {
      this.outputBias = this.addWeight('output_bias', [this.outputDim], null, this.biasInitializer, undefined, true);
    } else {
      this.outputBias = null;
    }

    this.recurrentTree = this.addWeight(
      'recurrent_tree',
      [inputShape[inputShape.length - 1] + this.stateSize, this.stateSize],
      null,
      this.recurrentInitializer,
      undefined,
      true
    );
    if (this.useRecurrentBias) {
      this.recurrentBias = this.addWeight(
        'recurrent_bias',
        [this.stateSize],
        null,
        this.biasInitializer,
        undefined,
        true
      );
    } else {
      this.recurrentBias = null;
    }
    this.built = true;
  }

  // Porting Note: PyKeras' equivalent of this method takes two tensor inputs:
  //   `inputs` and `states`. Here, the two tensors are combined into an
  //   `Tensor[]` Array as the first input argument.
  //   Similarly, PyKeras' equivalent of this method returns two values:
  //    `output` and `[output]`. Here the two are combined into one length-2
  //    `Tensor[]`, consisting of `output` repeated.
  call(inputs: Tensor | Tensor[], _kwargs: Kwargs): Tensor | Tensor[] {
    return tidy(() => {
      inputs = inputs as Tensor[];
      if (inputs.length !== 2) {
        throw new Error(`SimpleRNNCell expects 2 input Tensors, got ${inputs.length}.`);
      }
      const prevOutput = inputs[1];
      inputs = inputs[0];

      // Combine inputs and previous state into a single tensor.
      const combinedInputs = K.concatenate([inputs, prevOutput], 1);

      let output = K.dot(combinedInputs, this.outputTree.read());
      if (this.outputBias != null) {
        output = K.biasAdd(output, this.outputBias.read());
      }
      if (this.outputActivation != null) {
        output = this.outputActivation.apply(output);
      }

      let newState = K.dot(combinedInputs, this.recurrentTree.read());
      if (this.recurrentBias != null) {
        newState = K.biasAdd(newState, this.recurrentBias.read());
      }
      if (this.recurrentActivation != null) {
        newState = this.recurrentActivation.apply(newState);
      }

      return [output, newState];
    });
  }
}

export interface MyRNNLayerArgs extends RNNLayerArgs {
  cell: MySimpleRNNCell | MySimpleRNNCell[];

  trainableInitialState?: boolean;
  initialStateActivation?: ActivationIdentifier | null;
}

export class MyRNN extends tf.RNN {
  private outputDim: number;
  public readonly cell: RNNCell;

  readonly trainableInitialState: boolean;
  readonly initialStateActivation: Activation | null;

  public initialStateValues?: LayerVariable | LayerVariable[];

  constructor(args: MyRNNLayerArgs) {
    super(args);
    this.outputDim = Array.isArray(args.cell) ? args.cell[args.cell.length - 1].outputDim : args.cell.outputDim;

    if (Array.isArray(args.cell)) {
      this.cell = new MyStackedRNNCells({ cells: args.cell });
    }

    this.trainableInitialState = args.trainableInitialState || false;
    this.initialStateActivation = args.initialStateActivation ? getActivation(args.initialStateActivation) : null;
  }

  public build(inputShape: Shape | Shape[]): void {
    super.build(inputShape);

    if (this.trainableInitialState) {
      const buildInitialStateValues = (stateSize: number, i?: number) =>
        this.addWeight(
          `initial_state_weights${i == null ? '' : `_${i}`}`,
          [stateSize],
          undefined,
          getInitializer('glorotNormal'),
          undefined,
          true
        );

      const stateSize = this.cell.stateSize;
      if (Array.isArray(stateSize)) {
        this.initialStateValues = stateSize.map((stateSize, i) => buildInitialStateValues(stateSize, i));
      } else {
        this.initialStateValues = buildInitialStateValues(stateSize);
      }
    }
  }

  computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
    if (isArrayOfShapes(inputShape)) {
      inputShape = (inputShape as Shape[])[0];
    }
    inputShape = inputShape as Shape;

    let stateSize = this.cell.stateSize;
    if (!Array.isArray(stateSize)) {
      stateSize = [stateSize];
    }
    let outputShape: Shape | Shape[];
    if (this.returnSequences) {
      outputShape = [inputShape[0], inputShape[1], this.outputDim];
    } else {
      outputShape = [inputShape[0], this.outputDim];
    }

    if (this.returnState) {
      const stateShape: Shape[] = [];
      for (const dim of stateSize) {
        stateShape.push([inputShape[0], dim]);
      }
      return [outputShape].concat(stateShape);
    } else {
      return outputShape;
    }
  }

  getInitialState(inputs: Tensor): Tensor[] {
    return tidy(() => {
      if (this.initialStateValues) {
        if (Array.isArray(this.initialStateValues) && this.initialStateValues.length > 1) {
          throw new Error('MyRNNCell accepts only a single initial state value.');
        }

        // console.log('inputs shape: ', inputs.shape);
        let initialState = Array.isArray(this.initialStateValues)
          ? this.initialStateValues[0].read()
          : this.initialStateValues.read();
        // console.log('initialState shape: ', initialState.shape);
        initialState = K.expandDims(initialState, 0);
        // console.log('initialState shape: ', initialState.shape);
        initialState = K.tile(initialState, [inputs.shape[0], 1]);
        // console.log('initialState shape: ', initialState.shape);
        if (this.initialStateActivation) {
          initialState = this.initialStateActivation.apply(initialState);
        }

        return [initialState];
      }

      // Build an all-zero tensor of shape [samples, outputDim].
      // [Samples, timeSteps, inputDim].
      let initialState: tf.Tensor<tf.Rank> = tf.zeros(inputs.shape);
      // [Samples].
      initialState = tf.sum(initialState, [1, 2]);
      initialState = K.expandDims(initialState); // [Samples, 1].

      if (Array.isArray(this.cell.stateSize)) {
        return this.cell.stateSize.map((dim) => (dim > 1 ? K.tile(initialState, [1, dim]) : initialState));
      } else {
        return this.cell.stateSize > 1 ? [K.tile(initialState, [1, this.cell.stateSize])] : [initialState];
      }
    });
  }

  // getWeights(): Tensor[] {
  // 	const weights = super.getWeights();
  // 	if (!this.initialStateValues) {
  // 		return weights;
  // 	}

  // 	const initialStateValues = Array.isArray(this.initialStateValues)
  // 		? this.initialStateValues
  // 		: [this.initialStateValues];
  // 	console.log(initialStateValues);
  // 	return weights.concat(initialStateValues.map((weight) => weight.read()));
  // }

  get trainableWeights(): LayerVariable[] {
    if (!this.trainable) {
      return [];
    }
    // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
    return this.cell.trainableWeights.concat(this.initialStateValues ?? []);
  }
}
