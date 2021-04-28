var eps = 0.02;

matrixFactorization();

function matrixFactorization(numEpochs=1000,debugCount=1) {
  let W = [ [1, 3, 5, -2], 
			[2, 4, -3, -1], 
			[-1, 6, 3, -5] ]; // W:3x4
            
  let U = createVectors(3, 5), V = createVectors(5,4); //U:3x5,V:5x4
  let lastU = 0, lastV = 0;
  let patience = 0, lastLoss = 0;
  
  for(let epoch = 1; epoch <=numEpochs; epoch++) {
    //  forward
    let _W = multiply(U, V);
    let E = substract(_W, W); //  E: 3x4
    let loss = norm2(E);
    console.log('epoch: ' + epoch + ' loss = ' + loss);

    //  backward
	update(U, multiply(E, transpose(V)), lastU);	//  U:3x5 E*V.T:3x5
	update(V, multiply(transpose(U), E), lastV);	//  V:5x4 U.T*E:5x4
	
		
    // early stopping
    if(lastLoss && loss > lastLoss) {
      patience += 1;
      if(patience >= debugCount) {
        console.log('Earlystop at epoch ' + epoch + '/' + numEpochs);
        break;
      }
    }
    else patience = 0;
    lastLoss = loss;
    //	square error changes
	lastU = multiply(E, transpose(V));
  	lastV = multiply(transpose(U), E);
  }
  display(U, 'U')
  display(V, 'V')
  display(W, 'W')
  display(multiply(U,V), 'U*V')
}


function display(M, id) {
  console.log('------ '+id+' ------');
  M.forEach(arr => {
    console.log(arr.map(v=>(''+v).substring(0,8)).reduce((a,b)=>a+'\t'+b))
  })
}

function update(U, _U, __U) {
  for(let i = 0; i < U.length; i++) {
    for(let j = 0; j < U[i].length; j++) {
      if(__U && _U[i][j]*__U[i][j]>0) _U[i][j] += __U[i][j];
      U[i][j] -= eps * _U[i][j];
    }
  }
}

function norm2(E) {
  let sum = 0;
  for(let i = 0; i < E.length; i++) {
    for(let j = 0; j < E[i].length; j++) {
      sum += E[i][j] * E[i][j];
    }
  }
  return sum;
}

function multiply(A, B) {
  let C = new Array(A.length);
  for(let i = 0; i < C.length; i++) C[i]=new Array(B[0].length).fill(0);
  for(let i = 0; i < C.length; i++) {
    for(let j = 0; j < C[i].length; j++) {
      for(let k = 0; k < B.length; k++) C[i][j]+= A[i][k]*B[k][j];
    }
  }
  return C;
}

function substract(A,B) {

  let C = new Array(A.length);
  for(let i = 0; i < C.length; i++) C[i]=new Array(A[0].length);
  for(let i = 0; i < C.length; i++) {
    for(let j = 0; j < C[i].length; j++) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }
  return C;
}

function transpose(A) {
  let C = new Array(A[0].length);
  for(let i = 0; i < C.length; i++) {
    C[i]=new Array(A.length);
    for(let j = 0; j < C[i].length; j++) {
      C[i][j] = A[j][i];
    }
  }
  return C;
}

function createVectors(m, n) {
  let A = new Array(m);
  for(let i = 0; i < A.length; i++) {
    A[i]=createVector(n);
  }
  return A;
  
  function createVector(num) {
    let arr = new Array(num);
    for(let i = 0; i < arr.length; i++) {
      arr[i] = 0.1*(Math.random()-0.5);
    }
    return arr;
  }
}