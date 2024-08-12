const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const bestScore = document.getElementById("best_score");
const scoreCounter = document.getElementById("score");
const gameOver = document.getElementById("game_over");
const epochCounter = document.getElementById("epoch");
const reload = document.getElementById("reload");
const play = document.getElementById("play");
const epochContainer = document.getElementById("epoch_container");

reload.addEventListener("click", () => {
  try{
    localStorage.removeItem('best_dino');
    localStorage.removeItem('epoch_dino');
    localStorage.removeItem('allRewards');
    localStorage.removeItem('allGradients');
    localStorage.removeItem('tensorflowjs_models/model_dinosAIr/weight_specs');
    localStorage.removeItem('tensorflowjs_models/model_dinosAIr/info');
    localStorage.removeItem('tensorflowjs_models/model_dinosAIr/model_topology');
    localStorage.removeItem('tensorflowjs_models/model_dinosAIr/model_metadata');
    localStorage.removeItem('tensorflowjs_models/model_dinosAIr/weight_data');
    location.reload();
  } catch(error) {
    console.log('No history in ls')
  }

})


canvas.width = 1000;
canvas.height = 400;
let gameSpeed = 14;
// how often the obstacles and clouds appear 
let objectsFrequency = 1000;
const numOfSignals = 6;
const nHidden = 4;
const nHidden2 = 4;
const nOutputs = 1;
const learningRate = 0.02;
const discountRate = 0.90;
// let epoch = 1;

// const model = await tf.loadLayersModel('localstorage://my-model');
// console.log(model)

// async function loadModel() {
//   // let model;
//   try {
//     // Wczytanie modelu z LocalStorage
//     const model = await tf.loadLayersModel('localstorage://my-model');
//     console.log('Model loaded from LocalStorage');
//     model.summary();  // Wyświetlenie podsumowania modelu
//   } catch (error) {
//     console.error('Error loading model from LocalStorage:', error);

//     // nn definition 
//     const model = await tf.sequential();
//     model.add(tf.layers.dense({
//       units: nHidden,
//       activation: 'elu',
//       inputShape: [numOfSignals],
//       kernelInitializer: 'varianceScaling'
//     }));
//     model.add(tf.layers.dense({
//       units: nOutputs,
//       activation: 'sigmoid'
//     }));

//     // compilation 
//     let optimizer = tf.train.adam(learningRate);
//     model.compile({
//       optimizer: optimizer,
//       loss: 'binaryCrossentropy'
//     });
//     console.log(model)
//     // return model;
//     animate()
//   }
// }


// Funkcja do zapisu modelu do local storage
async function saveModel(model) {
  await model.save(`localstorage://model_dinosAIr`);
}

// Funkcja do wczytania modelu z local storage
async function loadModel() {
  return await tf.loadLayersModel(`localstorage://model_dinosAIr`);
}

// Funkcja do stworzenia nowego modelu
function createModel() {
  // nn definition 
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: nHidden,
    activation: 'elu',
    inputShape: [numOfSignals],
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.dense({
    units: nHidden2,
    activation: 'elu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.dense({
    units: nOutputs,
    activation: 'sigmoid'
  }));

  // compilation 
  // optimizer = tf.train.adam(learningRate);
  model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy'
  });
  // console.log('siema',model)
  return model
}

let model;
let optimizer;
async function main() {
  optimizer = tf.train.adam(learningRate);
  try {
      // Sprawdzenie, czy model jest w local storage i próba jego wczytania
      model = await loadModel();
      console.log('Model załadowany z local storage');
  } catch (error) {
      // Jeśli nie ma modelu w local storage, utworzenie nowego modelu
      console.log('Model nie znaleziony w local storage, tworzę nowy model');
      model = createModel();
      // (Opcjonalnie) trenowanie modelu tutaj, a następnie zapisanie go do local storage
      // await model.fit(data, labels);
      // await saveModel(model);
      // console.log('Nowy model zapisany do local storage');
  }
  for (let i = 0; i < model.getWeights().length; i++) {
    console.log(model.getWeights()[i].dataSync());
  }

  animate()
}

main()


//loss function 
function lossFunction(labels, logits) {
  return tf.losses.sigmoidCrossEntropy(labels, logits);
}

// discount reward function 
function discountRewards(rewards, discountRate) {
  const discountedRewards = [];
  let cumulativeRewards = 0;
  for (let i = rewards.length - 1; i >= 0; i--) {
    cumulativeRewards = rewards[i] + cumulativeRewards * discountRate;
    discountedRewards[i] = cumulativeRewards;
  }
  return discountedRewards;
}

// discount and normalize rewards function
function discountAndNormalizeRewards(allRewards, discountRate) {
  const allDiscountedRewards = allRewards.map(rewards => discountRewards(rewards, discountRate));
  const flatRewards = allDiscountedRewards.flat();
  const rewardMean = tf.mean(flatRewards);
  const rewardStd = tf.moments(flatRewards).variance.sqrt();
  return allDiscountedRewards.map(discountedRewards =>
    tf.tensor(discountedRewards).sub(rewardMean).div(rewardStd).arraySync()
  );
}

function sampleAction(prob) {
  return Math.random() < prob ? 1 : 0;
}

// function scaleAndAverageTensors(tensors, scaledRewards) {
//   const scaledTensors=[]
//   // console.log(tensors)
//   return tf.tidy(() => {
//     for(let i=0;i<tensors.length;i++){
//       scaledTensors[i] = tensors[0][i].value.mul(scaledRewards[i]);
//     }
    

//     // Sumowanie przeskalowanych tensorów
//     const sum = scaledTensors.reduce((acc, tensor) => acc.add(tensor));

//     // Obliczanie średniej
//     const average = sum.div(tensors.length);

//     // Zwracanie średniego tensoru
//     return average;
//   });
// }

function scaleAndAverageGrads(gradObjects, scaledRewards) {
  return tf.tidy(() => {
    const gradNames = Object.keys(gradObjects[0][0].grads);

    const scaledGradsList = []
    for(let i=0;i<scaledRewards.length;i++){
      // Przemnożenie gradientów przez liczbę

        // console.log(gradObjects[0][i].grads[gradNames[0]].mul(scaledRewards[0][i]))
        const scaledGrads = {};
        for (const name of gradNames) {
          scaledGrads[name] = gradObjects[0][i].grads[name].mul(scaledRewards[0][i]);
        }
        scaledGradsList[i] = scaledGrads
    }
    // console.log(scaledGradsList)

    // Sumowanie przeskalowanych gradientów
    const sumGrads = gradNames.reduce((acc, name) => {
      acc[name] = scaledGradsList.reduce((acc, scaledGrads) => {
        return acc.add(scaledGrads[name]);
      }, tf.zerosLike(gradObjects[0][0].grads[name]));
      return acc;
    }, {});

    // Obliczanie średniej
    const avgGrads = {};
    for (const name of gradNames) {
      avgGrads[name] = sumGrads[name].div(gradObjects.length);
    }

    return avgGrads;
  });
}

// Function to apply gradients
// function applyGradients(normalizedRewards, allGradients) {
//   tf.tidy(() => {
//     // Calculate the mean gradients weighted by the normalized rewards
//     const meanGradients = {};
//     for (let i = 0; i < allGradients[0].length; i++) {
//       meanGradients[i] = tf.zerosLike(allGradients[0][i]);
//     }

//     for (let i = 0; i < allGradients.length; i++) {
//       const reward = tf.tensor(normalizedRewards[i]);
//       for (let j = 0; j < allGradients[i].length; j++) {
//         meanGradients[j] = meanGradients[j].add(allGradients[i][j].mul(reward));
//       }
//     }

//     for (let i = 0; i < allGradients[0].length; i++) {
//       meanGradients[i] = meanGradients[i].div(tf.scalar(allGradients.length));
//     }

//     optimizer.applyGradients(meanGradients);
//   });
// }

function getBest(best, start) {
  if(localStorage.getItem(best) === null){
      return start.padStart(6, "0")
  } else {
      return JSON.parse(localStorage.getItem(best)).toString().padStart(6, "0")
  }
}

function getFromLocal(item) {
  if(localStorage.getItem(item) === null){
      return 1
  } else {
      return JSON.parse(localStorage.getItem(item))
  }
}

bestScore.innerHTML = getBest("best_dino", "0");
epochCounter.innerHTML = getFromLocal('epoch_dino');
let gameState = getFromLocal('game_state')


//adjust interface based on game state
if(gameState == 1) {
  play.innerHTML = 'Want to play on your own?';
  epochContainer.style.color = 'gray';
  reload.style.display = 'flex';
} else {
  play.innerHTML = 'Get back to learining!';
  epochContainer.style.color = 'white';
  reload.style.display = 'none';
}

// change from learning to playing game state
play.addEventListener("click", () => {
  if(gameState == 1) {
    gameState = 0
    play.innerHTML = 'Get back to learining!'
  } else {
    gameState = 1
    play.innerHTML = 'Want to play on your own?'
  }
  localStorage.removeItem('best_dino');
  localStorage.setItem(
    "game_state",
    JSON.stringify(gameState)
  )
  location.reload()
})

// create game elements: dinosaur, obstacles, clouds...
class Dinosaur{
  constructor() {
    this.width = 80;
    this.height = 86;
    this.x = 50;
    this.y = canvas.height - this.height;
    this.up = false;
    // this.spaceUp = false;
    this.spaceDown = false;
    this.power = 0;
    this.speed = 0;
    this.maxJump = 2000;
    this.alive = true;
    this.color = "black";
    this.score = 0;
    this.currentImage = 0;
    this.images = ["img/dinosaur1.png", "img/dinosaur2.png"]
    this.image = new Image();
    this.image.src = this.images[this.currentImage];

    this.photoInterval = setInterval(() => {
      this.currentImage = (this.currentImage + 1) % this.images.length;
      this.image.src = this.images[this.currentImage];
    }, 200)

  }

  allowControls() {
    document.onkeydown=(e) => {
      if (e.code === "Space" && this.y+this.height===canvas.height) {
          this.spaceDown = true;
      }
    };
    document.onkeyup=(e) => {
      if (e.code === "Space") {
          // this.spaceUp = true;
          this.spaceDown = false;
          // this.up = true;
      }
    };
  }

  move(){
    if(this.alive){
      // on space down get power and release on space up, there is min and max power
      if(this.spaceDown  && this.y+this.height===canvas.height){
        // this.power = Math.min(Math.max(1000,this.power+=300),2100);
        this.power = this.maxJump;

      } 
      
    }
    // speed of jump is inversely proportional to the hight above ground
    let x = (this.y+this.height)/115
    this.speed = x*x*x
    // start falling when no more power left
    // switch(this.power>=2100 && this.spaceDown || (this.power>0 && !this.spaceDown)){
    switch(this.power>0){
      case true:
        this.y-=this.speed;
        this.power-=10*this.speed;
        this.spaceDown = false;
        break;
      case false:
        if(this.y+this.height+this.speed<canvas.height){
          this.y+=this.speed;
        }
        else {
          this.y = canvas.height-this.height
        }
        break;
    }
  }

  drawDinosaur() {
    // ctx.fillStyle = this.color;
    ctx.drawImage(this.image, this.x, this.y, this.width, this.height);
  }
}

//create dinosaur
dinosaur = new Dinosaur();
dinosaur.allowControls();

class Obstacle{
  constructor(type) {
    switch(type){
      case 1:
        this.width = 24;
        this.height = 50;
        this.color = "red";
        this.y = canvas.height-this.height;
        this.image = new Image();
        this.image.src = "img/obstacle1.png"
        break;
      case 2:
        this.width = 44;
        this.height = 90;
        this.color = "green";
        this.y = canvas.height-this.height;
        this.image = new Image();
        this.image.src = "img/obstacle2.png"
        break;
      case 3:
        this.width = 98;
        this.height = 80;
        this.color = "yellow";
        this.y = canvas.height-this.height-(Math.floor(Math.random() * 80) + 50);
        this.currentImage = 0;
        this.images = ["img/obstacle3a.png", "img/obstacle3b.png"]
        this.image = new Image();
        this.image.src = this.images[this.currentImage];
        this.photoInterval = setInterval(() => {
          this.currentImage = (this.currentImage + 1) % this.images.length;
          this.image.src = this.images[this.currentImage];
        }, 200)
        break;
      case 4:
        this.width = 139;
        this.height = 90;
        this.color = "yellow";
        this.y = canvas.height-this.height;
        this.image = new Image();
        this.image.src = "img/obstacle4.png"
        break;
      case 5:
        this.width = 116;
        this.height = 77;
        this.color = "yellow";
        this.y = canvas.height-this.height;
        this.image = new Image();
        this.image.src = "img/obstacle5.png"
        break;
      case 6:
        this.width = 77;
        this.height = 77;
        this.color = "yellow";
        this.y = canvas.height-this.height;
        this.image = new Image();
        this.image.src = "img/obstacle6.png"
        break;
    }
    this.x = canvas.width+Math.floor(Math.random() * 350);
  }

  drawObstacle() {
    // ctx.fillStyle = this.color;
    ctx.drawImage(this.image, this.x, this.y, this.width, this.height);
  }

  move() {
    this.x = this.x-gameSpeed;
  }
}

//create obstacles function and specify how often they eppear 
obstacles = []
function addObstacle(type){
    obstacles.push(new Obstacle(type))
}
const createObstacles = setInterval(() => {
  //random obstacle
  addObstacle(Math.floor(Math.random() * 6) + 1);
},
objectsFrequency);


// add points on the ground
const points = [];
function addPoint() {
  const x = canvas.width + Math.floor(Math.random() * 100);
  const y = canvas.height - Math.floor(Math.random() * 8);
  const l = Math.floor(Math.random() * 15);
  points.push({ x, y, l});
}
const createPoints = setInterval(addPoint, 60);

// add clouds
class Cloud{
  constructor(type){
    switch(type){
      case 1:
        this.x = canvas.width+Math.floor(Math.random() * 150);
        this.y = 50+Math.floor(Math.random() * 150);
        this.width = 130;
        this.height = 38;
        this.image = new Image();
        this.image.src = "img/cloud.png"
        break;
      case 2:
        this.x = canvas.width+Math.floor(Math.random() * 150);
        this.y = 50+Math.floor(Math.random() * 150);
        this.width = 100;
        this.height = 29;
        this.image = new Image();
        this.image.src = "img/cloud2.png"
        break;
    }
  }

  drawCloud() {
    ctx.drawImage(this.image, this.x, this.y, this.width, this.height);
  }

  move(){
    this.x-=gameSpeed/4;
  }
}

clouds = []
function addCloud(type){
    clouds.push(new Cloud(type))
}
const createClouds = setInterval(() => {
  addCloud(Math.floor(Math.random() * 2) + 1)
},
objectsFrequency*2);

class Ray{
  constructor(ax, ay, length, angle){
    this.ax = ax;
    this.ay = ay;
    this.length = length;
    this.angle = angle;
    this.bx = this.ax + this.length * Math.cos(this.angle*Math.PI/180);
    this.by = this.ay + this.length * Math.sin(this.angle*Math.PI/180);
  }

  updateRay(){
    this.ax = dinosaur.x+dinosaur.width;
    this.ay = dinosaur.y+20;
    this.bx = this.ax + this.length * Math.cos(this.angle*Math.PI/180);
    this.by = this.ay + this.length * Math.sin(this.angle*Math.PI/180);
  }

  drawRay(){
    // this.updateRay()
    // console.log(this.ax,this.ay,this.bx,this.by)
    ctx.beginPath();
    ctx.moveTo(this.ax, this.ay);
    ctx.lineTo(this.bx, this.by);
    ctx.stroke();
  }
}

//specify rays number and angle
rays = []
for(let i=-24;i<=8;i+=8){
  rays.push(new Ray(dinosaur.x+dinosaur.width, dinosaur.y+20,400,i));
}


//intersection functions 
function lerp(A,B,t) {
  return A+(B-A)*t;
}

function getIntersection(ax,ay,bx,by,cx,cy,dx,dy){
  const tTop = (dx-cx)*(ay-cy)-(dy-cy)*(ax-cx);
  const uTop = (cy-ay)*(ax-bx)-(cx-ax)*(ay-by);
  const bottom = (dy-cy)*(bx-ax)-(dx-cx)*(by-ay);

  if(bottom!=0){
      const t=tTop/bottom;
      const u=uTop/bottom;
      if(t>=0 && t<=1 && u>=0 && u<=1){
          return {
              // x:lerp(ax,bx,t),
              // y:lerp(ay,by,t),
              offset:t
          }
      }
  }

  return {offset:0};
}

//function for drawing objects
function draw(drawRays = false) {
  //line
  ctx.beginPath();
  ctx.moveTo(0, canvas.height-10);
  ctx.lineTo(canvas.width, canvas.height-10);
  ctx.stroke();
  //grass
  points.forEach(point => {
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    ctx.lineTo(point.x+point.l, point.y);
    ctx.stroke();
  })
  clouds.forEach(cloud => {
    cloud.drawCloud();
  })
  dinosaur.drawDinosaur();
  obstacles.forEach(obstacle => {
    obstacle.drawObstacle();
  });
  if(drawRays) {
      rays.forEach(ray => {
      ray.drawRay();
    })
  }
}

// Funkcja do zapisywania tablicy do local storage
function saveArrayToLocalStorage(array, key) {
  localStorage.setItem(key, JSON.stringify(array));
}

// Funkcja do wczytywania tablicy z local storage
function loadArrayFromLocalStorage(key) {
  let storedArray = localStorage.getItem(key);
  return storedArray ? JSON.parse(storedArray) : [];
}

const allRewards = loadArrayFromLocalStorage('allRewards');
// console.log(allRewards);
const allGradients = loadArrayFromLocalStorage('allGradients');
// console.log(allRewards);
const currentRewards = [];
const currentGradients = [];
let reward = 0;
//animation function
let signals = new Array(numOfSignals).fill(0);
function animate(){
  let reward = 0;
  //first signal as % jump hight above ground
  signals[0] = (canvas.height-dinosaur.y-dinosaur.height)/(dinosaur.maxJump/10)


    let actions = []
    const grads = tf.variableGrads(() => {
      const preds = model.predict(
        tf.tensor2d(signals, [1, numOfSignals])
      );
      actions = preds.arraySync().map(sampleAction);
      const actionsTensor = tf.tensor2d(actions, [actions.length, 1]);
      const loss = tf.losses.sigmoidCrossEntropy(actionsTensor, preds);
      return loss
    })
    
    // optimizer.applyGradients(grads.grads)

    // const [actionVal, gradientsVal] = tf.tidy(() => {
    //   const logits = model.predict(tf.tensor2d(signals.flat(), [1, numOfSignals]));
    //   const actionProb = tf.concat([logits, tf.ones([1, 1]).sub (logits)], 1);
    //   const action = tf.multinomial(tf.log(actionProb), 1);
    //   const actionTensor = tf.tensor2d([action.arraySync()[0][0]], [1, 1]);
    //   const loss = tf.losses.sigmoidCrossEntropy(actionTensor, logits);
    //   const gradients = tf.variableGrads(() => loss);
    //   return [action.arraySync()[0][0], gradients.grads];
    // });
  if(dinosaur.alive){

    //clear canvas on every frame and draw all elements
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    draw(false);
    dinosaur.score += 1;



    //if gameState is 1 - move dinosaur based on nn prediction
    if(gameState == 1) {
      dinosaur.spaceDown = actions[0];
    }
    dinosaur.move();

    //move points and remove when passed
    for (let i = points.length - 1; i >= 0; i--) {
      const point = points[i];
      // if(dinosaur.alive) {
        point.x -= gameSpeed;
      // } 
      // else {
      //   clearInterval(createPoints);
      // }
      if (point.x+point.l < 0) {
          points.splice(i, 1);
      } 
    }
    //move clouds and remove when passed
    for (let i = clouds.length - 1; i >= 0; i--) {
      const cloud = clouds[i];
      if(dinosaur.alive){
        cloud.move();
      } 
      // else {
      //   clearInterval(createClouds)
      // }
      if (cloud.x+cloud.width < 0) {
          clouds.splice(i, 1);
      } 
    }
    
    //check for collisions and set actions if dead
    // obstacles.forEach(obstacle => {
    //   if(obstacle.x+obstacle.width>dinosaur.x && obstacle.x<dinosaur.x+dinosaur.width && obstacle.y<dinosaur.y+dinosaur.height && obstacle.y+obstacle.height>dinosaur.y) {
    //     clearInterval(createObstacles);
    //     dinosaur.alive = false;
    //     dinosaur.color = "gray";
    //     clearInterval(dinosaur.photoInterval);
    //   }
    //   if(dinosaur.alive){
    //     obstacle.move();
    //   }
    // })

    //update position of rays
    rays.forEach(ray => {
      ray.updateRay();
    })



    for (let i = obstacles.length - 1; i >= 0; i--) {
      const obstacle = obstacles[i];
      if(obstacle.x+obstacle.width>dinosaur.x && obstacle.x<dinosaur.x+dinosaur.width && obstacle.y<dinosaur.y+dinosaur.height && obstacle.y+obstacle.height>dinosaur.y) {
        // clearInterval(createObstacles);
        dinosaur.alive = false;
      }
      // if(dinosaur.alive){
        obstacle.move();
      // }
      if (obstacle.x+obstacle.width < 0) {
        obstacles.splice(i, 1);
      }

      signals = []
      signals.push(0)
      rays.forEach(ray =>{
        const firstI = getIntersection(ray.bx,ray.by,ray.ax,ray.ay,obstacle.x,obstacle.y,obstacle.x,obstacle.y+obstacle.height);
        const secondI = getIntersection(ray.bx,ray.by,ray.ax,ray.ay,obstacle.x,obstacle.y,obstacle.x+obstacle.width,obstacle.y);
        const thirdI = getIntersection(ray.bx,ray.by,ray.ax,ray.ay,obstacle.x+obstacle.width,obstacle.y,obstacle.x+obstacle.width,obstacle.y+obstacle.height);
        signals.push(Math.max(firstI.offset,secondI.offset,thirdI.offset))
      })


      //reward for jumping above obstacle
      if(obstacle.x+obstacle.width >= dinosaur.x-10 && obstacle.x+obstacle.width <= dinosaur.x+10 && obstacle.y > dinosaur.y+dinosaur.height) {
        reward += 50;
      }
      
    }

    //update score
    scoreCounter.innerHTML = dinosaur.score.toString().padStart(6, "0");
    if(dinosaur.score%500 === 0) {
      gameSpeed += 1;
    }

    //reward if dinosaur is alive
    // const reward = 0;

  } 

  if(dinosaur.alive) {
    if(dinosaur.y+dinosaur.height===canvas.height) {
      reward += 1
    } else {
      reward += -3
    }
    currentRewards.push(reward);
    currentGradients.push(grads);
    requestAnimationFrame(animate);
  } else {
    //end game
    if(dinosaur.score > bestScore.innerHTML) {
      localStorage.setItem(
        "best_dino",
        JSON.stringify(dinosaur.score)
      )
    }
    clearInterval(createObstacles);
    clearInterval(createPoints);
    clearInterval(dinosaur.photoInterval);
    clearInterval(createClouds);
    gameOver.style.display = "flex";
    setTimeout(() => {
      document.onkeydown=(e) => {
        if (e.code === "Space") {
          location.reload();
        }
      }
    }, 1000);

    const newEpoch = Number(epochCounter.innerHTML)+1 
    if(newEpoch > getFromLocal('epoch_dino')) {
      localStorage.setItem(
        "epoch_dino",
        JSON.stringify(newEpoch)
      )
    }

    // if game over and gameState is 1 (AI learning) compute gradients and go to new epoch
    if(gameState == 1) {
      console.log('tutaj', allRewards)
      reward += -50;
      currentRewards.push(reward);
      currentGradients.push(grads);

      //saving rewards and gradients for next iteration
      allRewards.push(currentRewards);
      allGradients.push(currentGradients);
      // saveArrayToLocalStorage(allRewards, 'allRewards');
      // saveArrayToLocalStorage(allGradients, 'allGradients');
      //adjusting weights
      const normalizedRewards = discountAndNormalizeRewards(allRewards, discountRate);
      // applyGradients(normalizedRewards, allGradients);

      // console.log(allGradients)
      // console.log(allRewards)
      // console.log(normalizedRewards)

      // console.log(scaleAndAverageTensors(allGradients,normalizedRewards).grads);
      // optimizer.applyGradients(scaleAndAverageTensors(allGradients,normalizedRewards).grads);


      optimizer.applyGradients(scaleAndAverageGrads(allGradients, normalizedRewards));
      for (let i = 0; i < model.getWeights().length; i++) {
        console.log(model.getWeights()[i].dataSync());
      }
      saveModel(model);
      location.reload();
    }
  }

}



