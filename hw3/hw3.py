from statistics import covariance
import numpy as np

# you must use python 3.6, 3.7, 3.8, 3.9 for sourcedefender
import sourcedefender
from HomeworkFramework import Function



class optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func = target_func
        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        #initialization
        #user defined input parameters
        self.mean = np.random.uniform(self.lower, self.upper, self.dim)
        self.sigma = 0.3*np.max(self.upper-self.lower)

        #strategy parameter setting: selection
        self.lambda_ = 4 + int(np.floor(3 * np.log(self.dim)))
        self.mu = int(np.floor(self.lambda_ / 2))
        self.weights = [np.log(self.mu + 1 / 2) - np.log(i) for i in range(1, self.mu + 1)]
        self.weights = self.weights / sum(self.weights)
        self.mueff = np.square(sum(self.weights)) / sum([np.square(i) for i in self.weights])

        #strategy parameter setting: adaptation
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / (np.square(self.dim + 1.3) + self.mueff)
        self.cmu = 2 * ((self.mueff - 2 + 1 / self.mueff) / (np.square(self.dim + 2) + 2 * self.mueff / 2))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        #initialize dynamic strategy parameters and constatnts
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.identity(self.dim)
        self.D = np.ones(self.dim)
        self.D_1 = self.B @ np.diag(1 / self.D) @ self.B.T
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * np.square(self.dim)))

        self.parents = np.full(self.mu, None)
        self.offsprings = np.full(self.lambda_, None)
        self.values = np.full(self.lambda_, None)
        self.mean_ori = np.copy(self.mean)
        self.hsig = 1
        self.z = np.zeros(self.dim)
        self.counteval = 0

        while self.eval_times < FES:
            #generate and evaluate lambda offspring
            for i in range(self.lambda_):
                self.offsprings[i] = self.mean + self.sigma * self.B @ (self.D * np.random.randn(self.dim))
                np.clip(self.offsprings[i], self.lower, self.upper, out=self.offsprings[i])
            for i in range(self.lambda_):
                value = self.f.evaluate(func_num, self.offsprings[i])
                self.eval_times += 1
                if value == "ReachFunctionLimit":
                    print("ReachFunctionLimit")
                    break
                self.values[i] = value

            #sort by fitness and update mean
            rank = np.argsort(self.values)
            if self.values[rank[0]] < self.optimal_value:
                self.optimal_solution[:] = self.offsprings[rank[0]]
                self.optimal_value = self.values[rank[0]]
            self.parents[:] = self.offsprings[rank[:self.mu]]
            self.mean_ori[:] = self.mean
            m = np.zeros(self.dim)
            for i in range(len(self.parents)):
                m += self.parents[i] * self.weights[i]
            self.mean[:] = m

            #cumulation: update evolution paths
            self.z = (self.mean - self.mean_ori) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.D_1 @ self.z
            self.hsig = np.linalg.norm(self.ps) / np.sqrt(1 - np.power((1 - self.cs), 2 * self.counteval / self.lambda_ + 1)) / self.chiN < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + self.hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * self.z

            #adapt covariance matrix C
            x = np.empty((self.mu, self.dim))
            for i in range(len(x)):
                x[i] = self.parents[i]
            artmp = (1 / self.sigma) * (x - np.tile(self.mean_ori, (self.mu, 1)))
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - self.hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * (artmp.T @ np.diag(self.weights) @ artmp)
            
            #adapt step-size sigma
            self.sigma = self.sigma * np.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / self.chiN - 1))

            #update B and D from C 
            self.C = np.triu(self.C) + np.triu(self.C, k=1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D)
            self.D_1 = self.B @ np.diag(1 / self.D) @ self.B.T

            #update counteval
            self.counteval += 1

            # print("optimal: {}\n".format(self.get_optimal()[1]))

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
