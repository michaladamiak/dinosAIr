class Network {
    constructor(inputs, hidden, outputs, learningRate) {
        this.inputs = inputs;
        this.hidden = hidden;
        this.outputs = outputs;
        this.learningRate = learningRate;
        this.X = tf.placeholder(tf.float32, shape=[NaN, inputs])
    }
    test() {
        console.log('siema')
    }
}