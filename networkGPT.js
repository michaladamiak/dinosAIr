class NeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // Inicjalizacja wag
    this.weights1 = tf.variable(tf.randomNormal([this.inputSize, this.hiddenSize]));
    this.weights2 = tf.variable(tf.randomNormal([this.hiddenSize, this.outputSize]));
  }

  predict(input) {
    const inputTensor = tf.tensor(input, [1, this.inputSize]);
    const hidden = inputTensor.matMul(this.weights1).relu();
    const output = hidden.matMul(this.weights2).softmax();
    return output;
  }

  async train(states, actions, rewards, learningRate = 0.01) {
    // Inicjalizacja optymalizatora Adam z określoną szybkością uczenia
    const optimizer = tf.train.adam(learningRate);

    // Funkcja obliczająca stratę
    const computeLoss = () => {
      let loss = tf.scalar(0); // Inicjalizacja straty jako skalar z wartością 0

      // Iteracja przez wszystkie zgromadzone stany, akcje i nagrody
      for (let i = 0; i < states.length; i++) {
        const inputTensor = tf.tensor(states[i], [1, this.inputSize]); // Konwersja stanu do tensora
        const actionTensor = tf.tensor(actions[i], [1, this.outputSize]); // Konwersja akcji do tensora

        // Przepuszczenie wejścia przez sieć neuronową
        const hidden = inputTensor.matMul(this.weights1).relu(); // Mnożenie macierzy wejściowej przez wagi pierwszej warstwy i zastosowanie funkcji ReLU
        const output = hidden.matMul(this.weights2).softmax(); // Mnożenie macierzy wyjściowej ukrytej przez wagi drugiej warstwy i zastosowanie funkcji softmax

        const actionProb = output.mul(actionTensor).sum(); // Obliczanie prawdopodobieństwa wybranej akcji
        const logProb = actionProb.log(); // Obliczanie logarytmu prawdopodobieństwa wybranej akcji
        const rewardTensor = tf.scalar(rewards[i]); // Konwersja nagrody do tensora

        // Aktualizacja wartości straty
        loss = loss.add(logProb.mul(rewardTensor)); // Dodawanie do straty wartości logarytmu prawdopodobieństwa pomnożonej przez nagrodę
      }
      return loss.mul(tf.scalar(-1)); // Zwracanie negatywnej wartości straty (chcemy minimalizować stratę)
    };

    // Minimalizacja funkcji straty przy użyciu optymalizatora Adam
    optimizer.minimize(computeLoss);
  }
}

// Główna pętla gry
const inputSize = 10; // Przykładowy rozmiar wejścia
const hiddenSize = 16;
const outputSize = 2; // Skok lub brak skoku

const nn = new NeuralNetwork(inputSize, hiddenSize, outputSize);

// Przykładowa pętla gry
const states = [];
const actions = [];
const rewards = [];

// W trakcie każdej iteracji gry
function gameStep() {
  const state = getCurrentState();
  const actionProb = nn.predict(state).dataSync();
  const action = Math.random() < actionProb[0] ? 0 : 1; // 0: brak skoku, 1: skok
  performAction(action);

  const reward = getReward();
  states.push(state);
  actions.push([action === 0 ? 1 : 0, action === 1 ? 1 : 0]);
  rewards.push(reward);

  // Sprawdzenie końca gry
  if (isGameOver()) {
    // Obliczanie skumulowanych nagród
    const discountedRewards = [];
    let cumulativeReward = 0;
    for (let i = rewards.length - 1; i >= 0; i--) {
      cumulativeReward = rewards[i] + cumulativeReward * 0.99;
      discountedRewards.unshift(cumulativeReward);
    }

    // Trening sieci neuronowej
    nn.train(states, actions, discountedRewards).then(() => {
      // Restart gry
      resetGame();
    });
  } else {
    requestAnimationFrame(gameStep);
  }
}

// Rozpoczęcie gry
requestAnimationFrame(gameStep);

// Funkcje pomocnicze
function getCurrentState() {
  // Implementacja pobierania aktualnego stanu gry
}

function performAction(action) {
  // Implementacja wykonania akcji (skok/brak skoku)
}

function getReward() {
  // Implementacja pobierania nagrody
}

function isGameOver() {
  // Implementacja sprawdzenia, czy gra się zakończyła
}

function resetGame() {
  // Implementacja resetowania gry
}