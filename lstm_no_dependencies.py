class LSTM:
    t = 0 #number of times lstm has been unrolled/unfolded
    cell = [] #snapshots of lstm's values at each time step as it is being unrolled (needed for backprop through time)
    W,U,b,dW,dU,db = {},{},{},{},{},{} #LSTM parameters for learning (Weights and biases)

    #-----random number generator using cellular automata-----
    def rule30(self, a,b,c):
        if int(a) + int(b) + int(c) == 1:
            return '1'
        elif a != b == c == '1':
            return '1'
        return '0'

    def cellular_automata(self, rule, row):
        nextrow = ''
        row = row[-1] + row + row[0]
        for a,b,c in zip(row,  row[1:], row[2:]):
            nextrow += rule(a,b,c)
        return nextrow

    def bin2float(self, binary, bit_len):
        sign,n = [1,-1][int(binary[0])],int(binary[1:],2)
        return sign*n / 2**(bit_len)

    def random(self):
        self.seed = self.cellular_automata(self.rule30, self.seed)
        return self.bin2float(self.seed,12)
    
    def seed(self,x = 1):
        self.seed = '{0:013b}'.format(x)

    #-----sine / hyperbolic-tan functions for gate/neuron activation/squashing----
    def e(self, x):
        euler = 2.718281828459045235360287471352662
        x = max(min(x,20),-20)
        return euler**x

    def tanh(self,x):
        try:
            a,b = self.e(x),self.e(-x)
            return (a-b)/(a+b)
        except: #applying tanh element-wise to a matrix
            try:
                return [[self.tanh(n) for n in X] for X in x]
            except:
                return [self.tanh(X) for X in x]
        
    def sigmoid(self,x):
        try:
            return 1/(1 + self.e(-x))
        except: #applying sigmoid element-wise to a matrix
            try:
                return [[self.sigmoid(n) for n in X] for X in x]
            except:
                return [self.sigmoid(X) for X in x]
    
    # def dSigmoid(self,x):
    #     sx = self.sigmoid(x)
    #     return sx*(1-sx)

    def dTanh(self,x): #derivative of tanh
        try:
            return 1 - self.tanh(x) **2
        except: #applying sigmoid element-wise to a matrix
            try:
                return [[1 - self.tanh(n) **2 for n in X] for X in x]
            except:
                return [1 - self.tanh(X) **2 for X in x]
    
    def log(self, x,base=10,precision = 12):
        re = []
        for _ in range(precision):
            n = 0
            while base**n <= x:
                n+= 1
            n -= 1
            re.append(str(n))
            x /= base**n 
            x **= base
        return float(re[0] + '.' + ''.join(re[1:]))

    #-------matrix & vector functions-----------
    def print(self,mX):
        try: #print a matrix neatly
            for X in mX:
                row = '[ '
                for x in X:
                    if str(x)[0] == '-':
                        row += '{0:.1f}  '.format(x)
                    else:
                        row += ' {0:.1f}  '.format(x)
                print('\t' + row + ']')
        except:
            row = '[ '
            for x in mX:
                if str(x)[0] == '-':
                    row += '{0:.1f}  '.format(x)
                else:
                    row += ' {0:.1f}  '.format(x)
            print('\t' + row + ']')

    def zeros(self, topology): #empty matrix
        try:
            rows,columns = topology
            return [[.0 for _ in range(rows)] for _ in range(columns)]
        except:
            return [.0 for _ in range(topology)]

    def randoms(self, topology):  #a matrix filled with random floats
        try: 
            rows,columns = topology
            return [[self.random() for _ in range(rows)] for _ in range(columns)]
        except: 
            return [self.random() for _ in range(topology)]

    def tran(self,mX): # transpose
        try:
            return [[row[col] for row in mX] for col in range(len(mX[0]))]
        except:
            return [[x] for x in mX]
       
    def dot(self,mA,mB): # inner scalar product (NB: proper dot product dimensions = [x,a]@[b,x]  = [b,a]  BUT for simplicity, this dot products dimensions: [x,a]@[x,b] = [b,a] )
        try: #[[matrix]] @ [[matrix]]
            return [[sum(a*b for a,b in zip(A,B)) for B in mB] for A in mA]
        except:
            try: # [vector] @ [vector]
                return [sum(a*b for a,b in zip(mA,mB))]
            except: # [vector] @ (onehot_id)
                return [mA[mB]]
  
    def star(self,mA,mB): # element-wise hammard product
        try:#[[matrix]] * [[matrix]]
            return [[a*b for a,b in zip(A,B)] for A,B in zip(mA,mB)]
        except:
            try:# [vector] * [vector]
                return [A*B for A,B in zip(mA,mB)]
            except:# [vector] * (onehot_id)
                vector = self.zeros(len(mA))
                vector[mB] = mA[mB]
                return vector

    def scale(self,mX,factor): # multiply a single number to each element in matrix
        try:
            return [[x*factor for x in X] for X in mX]
        except:
            return [X*factor for X in mX]
    
    def squared(self,mX): # element-wise squaring
        try:
            return [[x**2 for x in X] for X in mX]
        except:
            return [x**2 for x in mX]

    def complement(self,mX): # 1 - x 
        try:
            return [[1 - x for x in X] for X in mX]
        except:
            return [1 - x for x in mX]
 
    def minus(self,mA,mB): # element-wise subtraction
        try:
            return [[a-b for a,b in zip(A,B)] for A,B in zip(mA,mB)]
        except:
            return [a-b for a,b in zip(mA,mB)]
 
    def plus(self,mA,mB): # element-wise addition
        try: #[[matrix]] * [[matrix]]
            return [[a+b for a,b in zip(A,B)] for A,B in zip(mA,mB)]
        except:
            try: # [vector] * [vector]
                return [A+B for A,B in zip(mA,mB)]
            except: # [vector] * (onehot_id)
                mA[mB] += 1.
                return mA

    def add(self,mX,factor): #add a single +/- value to entire matrix 
        try:
            return [[x + factor for x in X] for X in mX]
        except:
            return [x + factor for x in mX]

    def argmax(self,mX):
        return mX.index(max(mX))

    #------LSTM calculations----------------------
    def calc_gate(self,label,x):
        return self.plus( self.plus(self.dot(self.W[label],self.tran(x)) , self.dot(self.U[label], self.tran(self.cell[-2]['out']) ) ) , self.b[label] )

    def calc_state(self):
        return self.plus( self.star( self.cell[-1]['gate']['a'] , self.cell[-1]['gate']['i'])  , self.star( self.cell[-1]['gate']['f'] , self.cell[-2]['state'] ) )
    
    def calc_out(self):
        return self.star( self.tanh(self.cell[-1]['state']) , self.cell[-1]['gate']['o'] )

    #---------Network functions---------------------
    def __init__(self,topology):
        self.x_dim, self.h_dim, _, self.b_dim = topology # (input dimension/word vector length, hidden dimension, output dimension/word vector length, batch dimension) e.g.(30,512,30, 1)
        self.seed() #for random number generator
        for label in 'aifo':
            self.W[label] = self.randoms((self.x_dim, self.h_dim))
            self.U[label] = self.randoms((self.h_dim, self.h_dim))
            self.b[label] = self.randoms((self.b_dim, self.h_dim))
            self.dW[label] = self.zeros((self.x_dim, self.h_dim))
            self.dU[label] = self.zeros((self.h_dim, self.h_dim))
            self.db[label] = self.zeros((self.b_dim, self.h_dim))
        self.dOut =  self.zeros((self.b_dim,self.h_dim))

    def train(self,x,y):
        self.blankCell()
        for x_t in x.values():
            self.t += 1
            self.FPTT(x_t)    
        self.blankCell()
        for i in range(len(self.cell)-3,-1,-1):
            self.BPTT(y[i])
            self.SGD(x[i])
            self.t -= 1
        self.update()

    def blankCell(self):         
        self.cell.append( {
            'gate':{label:self.zeros((self.b_dim, self.h_dim)) for label in 'aiof'} , 
            'dGate':{label:self.zeros((self.b_dim, self.h_dim)) for label in 'aiof'} , 
            'state':self.zeros((self.b_dim,self.h_dim)), 
            'dState':self.zeros((self.b_dim,self.h_dim)), 
            'out': self.zeros((self.b_dim, self.h_dim)) 
            })  

    def FPTT(self,x): #(Forward Propogation)
        print("\n-----forward pass (t = {})-----".format(self.t))
        self.cell.append({ 
            'gate':{}, 
            'dGate': {label:self.zeros((self.b_dim, self.h_dim)) for label in 'aiof'}  , 
            'dState':self.zeros((self.b_dim,self.h_dim)) 
            } )
        self.cell[-1]['gate']['a'] = self.tanh( self.calc_gate('a',x) )
        print("a_{} = tanh(Wa_{} @ x_{} + Ua_{} @ y_{} + ba_{}) =".format(self.t, self.t, self.t, self.t, self.t - 1, self.t))
        self.print(self.cell[-1]['gate']['a'])
        for label in 'ifo':
            self.cell[-1]['gate'][label] = self.sigmoid(self.calc_gate(label,x))
            print("{}_{} = sigmoid(W{}_{} @ x_{} + U{}_{} @ y_{} + b{}_{}) =".format(label,self.t, label,self.t, self.t,label,self.t, self.t - 1, label, self.t))
            self.print(self.cell[-1]['gate'][label])
        self.cell[-1]['state'] = self.calc_state()
        print("s_{} = a_{} * i_{} + f_{} * s_{} =".format(self.t, self.t, self.t, self.t, self.t - 1))
        self.print(self.cell[-1]['state'] )
        self.cell[-1]['out'] = self.calc_out()
        print("^y_{} = tanh(s_{}) * o_{} =".format(self.t, self.t, self.t))
        self.print(self.cell[-1]['out'])
        #or Cross-Entropy-Loss, L = sum L_t   
        # L_t += -y_t*log(out_t)

    def BPTT(self,y):  #(Back Propogation Through Time - BPTT)
        print("\n------backward pass (t = {})--------".format(self.t))
        dL2 = self.add( self.cell[self.t]['out'] , -y[0][0] )
        L2 = self.scale( self.squared(dL2) , .5)
        print("Loss = L2_{} = (^y_{} - y_{})**2 / 2 =".format(self.t, self.t, self.t))
        self.print(L2)
        print("dL2_{} = ^y_{} - y_{} =".format(self.t, self.t, self.t))
        self.print(dL2)
        dY = self.plus( dL2 , self.dOut )
        print("dY_{} = dL2_{} + dOut_{} =".format(self.t, self.t, self.t + 1))
        self.print(dY)
        self.cell[self.t]['dState'] = self.plus( self.star( self.star( dY, self.cell[self.t]['gate']['o']) , self.dTanh(self.cell[self.t]['state']))  , self.star( self.cell[self.t + 1]['dState'] , self.cell[self.t + 1]['gate']['f'] ) )
        print("dS_{} = dY_{} * o_{} * dTanh(s_{}) + dS_{} * f_{} =".format(self.t, self.t, self.t, self.t, self.t + 1, self.t + 1))
        self.print(self.cell[self.t]['dState'])
        self.cell[-1]['dGate']['a'] = self.star( self.star( self.cell[self.t]['dState'] , self.cell[self.t]['gate']['i'])  , ( self.complement( self.squared(self.cell[self.t]['gate']['a'])) ) )
        print("dA_{} = dS_{} * i_{} * (1 - a_{} **2) =".format(self.t, self.t, self.t, self.t))
        self.print(self.cell[-1]['dGate']['a'])
        self.cell[-1]['dGate']['i'] = self.star( self.star( self.star(  self.cell[self.t]['dState'] , self.cell[self.t]['gate']['a'])  , self.cell[self.t]['gate']['i']) , self.complement(self.cell[self.t]['gate']['i']) )
        print("dI_{} = dS_{} * a_{} * i_{} * (1 - i_{}) =".format(self.t, self.t, self.t, self.t, self.t))
        self.print(self.cell[-1]['dGate']['i'])
        self.cell[-1]['dGate']['f'] = self.star( self.star( self.star( self.cell[self.t]['dState'] , self.cell[self.t-1]['state'])  , self.cell[self.t]['gate']['f'])  , self.complement(self.cell[self.t]['gate']['f'] ))
        print('dF_{} = dS_{} * s_{} * f_{} * (1 - f_{}) ='.format(self.t, self.t, self.t -1, self.t, self.t))
        self.print(self.cell[-1]['dGate']['f'])
        self.cell[-1]['dGate']['o'] = self.star( self.star( self.star( dY , self.tanh(self.cell[self.t]['state']) ) , self.cell[self.t]['gate']['o'] ) , self.complement(self.cell[self.t]['gate']['o'] ) )
        print('dO_{} = dY_{} * tanh(s_{}) * o_{} * (1 - o_{}) ='.format(self.t, self.t, self.t, self.t, self.t))
        self.print(self.cell[-1]['dGate']['o'])
        dX,dOut= {},{}
        for label in 'aifo': 
            dGateT = self.tran(self.cell[-1]['dGate'][label])
            dX[label] = self.dot(self.tran( self.W[label]), dGateT )
            print('dX{}_{} = W{}.T @ d{}_{}.T ='.format(label,self.t, label,label.upper(), self.t))
            self.print(dX[label] )  #dXa = WT.dGateT = [x,h]T . [b,h]T = [h,x].[h,b] = [b,x]
            dOut[label] = self.dot(self.tran(self.U[label]), dGateT )
            print('dOut{}_{} = U{}.T @ d{}_{}.T ='.format(label, self.t, label, label.upper(), self.t))
            self.print(dOut[label])
        dX = self.plus(dX['a'], self.plus(dX['i'],self.plus(dX['f'],dX['o'] ) ))
        print('sum(dX) =')
        self.print(dX) #dX = dXa + dXi + dXo + dXf = [b,x] + [b,x] + [b,x] + [b,x] = [b,x]
        self.dOut = self.plus(dOut['a'], self.plus(dOut['i'], self.plus( dOut['f'],dOut['o'] ) ))
        print('sum(dOut) =')
        self.print(self.dOut)   

    def SGD(self, x): #(Stochastic Gradient Descent - SGD)
        print("\n------Stochastic Gradient Descent --------")
        for label in 'aifo':
            self.dW[label] = self.plus(self.dW[label], self.dot(self.cell[self.t]['dGate'][label], x) )
            print('dW{} += dGate{}_{} @ x_{} ='.format(label, label,self.t, self.t))
            self.print(self.dW[label])
            self.dU[label] = self.plus(self.dU[label], self.dot(self.cell[self.t + 1]['dGate'][label] , self.cell[self.t]['out'] ))
            print('dU{} += dGate{}_{} @ ^y_{} ='.format(label, label, self.t + 1 , self.t))
            self.print(self.dU[label])
            self.db[label] = self.plus(self.db[label], self.cell[self.t]['dGate'][label])
            print('dB{} += dGate{}_{} ='.format(label, label,self.t))
            self.print(self.db[label])

    def update(self, learning_rate = .1): #update W,U,b
        print("\n-----Adjusting Weights---------")
        for label in 'aifo':
            self.W[label] = self.minus(self.W[label], self.scale( self.dW[label], learning_rate))
            print("W{} -= dW{} * learning_rate =".format(label, label))
            self.print(self.W[label])
            self.U[label] = self.minus(self.U[label], self.scale( self.dU[label], learning_rate))
            print("U{} -= dU{} * learning_rate =".format(label, label))
            self.print(self.U[label])
            self.b[label] = self.minus(self.b[label], self.scale( self.db[label], learning_rate))
            print("b{} -= dB{} * learning_rate =".format(label, label))
            self.print(self.b[label])

#TESTING
#training inputs
x = {}
x[0] = [[1.], [2.]]
x[1] = [[.5], [3.]]  
#training outputs/targets/labels
y = {}
y[0] = [[.5]]
y[1] = [[1.25]]
#init & train network
nnet = LSTM((2,2,1,1))
nnet.train(x,y)

# onehot1 = nnet.zeros((3,1))
# onehot1[0][2] = 1
# onehot2 = (2)

# b = nnet.randoms(3)
# a = [b]

# re1 = nnet.plus(a,onehot1)
# re2 = nnet.plus(b,onehot2)
# print(re1)
# print(re2)

#converting words to ints & vice versa
print(ord('a'))
print(chr(1114111))
