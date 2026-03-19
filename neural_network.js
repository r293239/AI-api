// neural_network.js - Neural network for conversation learning

class NeuralNetwork {
    constructor(inputSize, hiddenSizes, outputSize, learningRate = 0.01) {
        this.learningRate = learningRate;
        this.layers = [];
        
        // Build network
        let sizes = [inputSize, ...hiddenSizes, outputSize];
        for (let i = 0; i < sizes.length - 1; i++) {
            // Xavier initialization
            const limit = Math.sqrt(6.0 / (sizes[i] + sizes[i+1]));
            this.layers.push({
                weights: this.randomMatrix(sizes[i], sizes[i+1], -limit, limit),
                biases: new Array(sizes[i+1]).fill(0),
                // Adam optimizer parameters
                mWeights: this.zeros(sizes[i], sizes[i+1]),
                vWeights: this.zeros(sizes[i], sizes[i+1]),
                mBiases: new Array(sizes[i+1]).fill(0),
                vBiases: new Array(sizes[i+1]).fill(0)
            });
        }
        
        // Adam parameters
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 1e-8;
        this.t = 0;
    }
    
    randomMatrix(rows, cols, min, max) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                row.push(min + Math.random() * (max - min));
            }
            matrix.push(row);
        }
        return matrix;
    }
    
    zeros(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix.push(new Array(cols).fill(0));
        }
        return matrix;
    }
    
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    sigmoidDerivative(x) {
        return x * (1 - x);
    }
    
    relu(x) {
        return Math.max(0, x);
    }
    
    reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }
    
    softmax(x) {
        const maxVal = Math.max(...x);
        const expValues = x.map(val => Math.exp(val - maxVal));
        const sumExp = expValues.reduce((a, b) => a + b, 0);
        return expValues.map(val => val / sumExp);
    }
    
    forward(input, training = true) {
        let activation = input;
        this.activations = [input];
        this.zValues = [];
        
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            
            // Calculate z = W * a + b
            const z = this.matrixVectorMultiply(layer.weights, activation);
            for (let j = 0; j < z.length; j++) {
                z[j] += layer.biases[j];
            }
            this.zValues.push(z);
            
            // Apply activation (sigmoid for hidden, softmax for output)
            if (i < this.layers.length - 1) {
                activation = z.map(x => this.relu(x));
            } else {
                activation = this.softmax(z);
            }
            this.activations.push(activation);
        }
        
        return activation;
    }
    
    matrixVectorMultiply(matrix, vector) {
        const result = new Array(matrix.length).fill(0);
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }
    
    vectorMatrixMultiply(vector, matrix) {
        const result = new Array(matrix[0].length).fill(0);
        for (let j = 0; j < matrix[0].length; j++) {
            for (let i = 0; i < vector.length; i++) {
                result[j] += vector[i] * matrix[i][j];
            }
        }
        return result;
    }
    
    backward(input, target) {
        // Forward pass
        const output = this.forward(input);
        
        // Calculate output error (cross-entropy derivative)
        let delta = output.map((val, i) => val - target[i]);
        
        this.t++;
        
        // Backward pass through layers
        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];
            
            // Get input to this layer
            const layerInput = i === 0 ? input : this.activations[i];
            
            // Calculate gradients
            const weightGrad = this.outerProduct(delta, layerInput);
            const biasGrad = [...delta];
            
            // Adam optimizer updates
            for (let j = 0; j < layer.weights.length; j++) {
                for (let k = 0; k < layer.weights[j].length; k++) {
                    // Update biased moments
                    layer.mWeights[j][k] = this.beta1 * layer.mWeights[j][k] + 
                                          (1 - this.beta1) * weightGrad[j][k];
                    layer.vWeights[j][k] = this.beta2 * layer.vWeights[j][k] + 
                                          (1 - this.beta2) * (weightGrad[j][k] * weightGrad[j][k]);
                    
                    // Bias correction
                    const mCorrected = layer.mWeights[j][k] / (1 - Math.pow(this.beta1, this.t));
                    const vCorrected = layer.vWeights[j][k] / (1 - Math.pow(this.beta2, this.t));
                    
                    // Update weight
                    layer.weights[j][k] -= this.learningRate * mCorrected / 
                                           (Math.sqrt(vCorrected) + this.epsilon);
                }
            }
            
            // Update biases
            for (let j = 0; j < layer.biases.length; j++) {
                layer.mBiases[j] = this.beta1 * layer.mBiases[j] + (1 - this.beta1) * biasGrad[j];
                layer.vBiases[j] = this.beta2 * layer.vBiases[j] + (1 - this.beta2) * (biasGrad[j] * biasGrad[j]);
                
                const mCorrected = layer.mBiases[j] / (1 - Math.pow(this.beta1, this.t));
                const vCorrected = layer.vBiases[j] / (1 - Math.pow(this.beta2, this.t));
                
                layer.biases[j] -= this.learningRate * mCorrected / (Math.sqrt(vCorrected) + this.epsilon);
            }
            
            // Calculate delta for previous layer (if not first layer)
            if (i > 0) {
                delta = this.vectorMatrixMultiply(delta, layer.weights);
                for (let j = 0; j < delta.length; j++) {
                    delta[j] *= this.reluDerivative(this.zValues[i-1][j]);
                }
            }
        }
    }
    
    outerProduct(vec1, vec2) {
        const result = [];
        for (let i = 0; i < vec1.length; i++) {
            const row = [];
            for (let j = 0; j < vec2.length; j++) {
                row.push(vec1[i] * vec2[j]);
            }
            result.push(row);
        }
        return result;
    }
    
    predict(input) {
        return this.forward(input, false);
    }
}
