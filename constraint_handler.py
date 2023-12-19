from pyscipopt import Model, Conshdlr, quicksum, multidict, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING
import numpy as np
from intersection_cuts import Intersection_Cuts

class Constraint_Handler(Conshdlr):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def FMILP(self, HPR_Solution_X ):

        problem_data, XDim, YDim = self.data
        [A, B, C, Dy]            = problem_data[-5:-1]

        m = Model("General Binary Bilevel Solver")
        Y_ = { f"{i}":m.addVar(vtype="B", name=f"Y{i}") for i in range(YDim) }

        m.setObjective( quicksum(Dy[idx]*Y_[f"{idx}"] for idx in range(YDim)), 'minimize' )

        num_follower_cons_rows = len(A)
        for row in range(num_follower_cons_rows):
            m.addCons( quicksum( A[row][idx]*HPR_Solution_X[idx] for idx in range(XDim)) + quicksum( B[row][idx]*Y_[f'{idx}'] for idx in range(YDim)) <= C[row] )

        try:
            m.optimize()
            OptY_Sol = [m.getVal(Y_[f"{idx}"]) for idx in range(YDim)]
            return OptY_Sol, m.getObjVal()
        except:
            return "Problem in FMILP",""

    def addcut(self, checkonly, sol):

            problem_data, XDim, YDim = self.data
            [A, B, C, Dy]            = problem_data[-5:-1]
            # Getting the solutions at current node
            X_sol, Y_sol = [], []
            for idx in range(XDim):
                xt   = self.model.getSolVal(sol, self.X[f'{idx}'])
                X_sol.append(xt)

            for idx in range(YDim):
                yt = self.model.getSolVal(sol, self.Y[f"{idx}"])
                Y_sol.append(yt)

            # print("Current Solution = ", X_sol, Y_sol,"\n")
            # Getting Optimal Y for current X solution.
            Optimal_Y, FMILP_Objective = self.FMILP( X_sol )
            # Checking Cut Violation (d @ HPRy* - phi(x*)[FMILP_Obj] > 0), checking bilevel feasibility
            # print(Y_sol, Dy @ np.array(Y_sol) )
            # print(Optimal_Y, Dy @ np.array(Optimal_Y))
            ConsCheck = Dy @ np.array(Y_sol) - FMILP_Objective
            # print(ConsCheck, checkonly, self.model.getSolObjVal(sol) )
            # print(self.model.getNLPCols(), self.model.getNLPRows(), self.model.getNVars() )
            # Adding Cut
            cutsadded  = False
            if ConsCheck > 0: # => Bilevel Infeasible Solution
                if checkonly:
                    return True # Will cut be added for this solution
                else:
                    # Getting Variable Basis Info
                    BasisInfo = self.model.getLPBasisInd() # "Gets all indices of basic columns and rows: index i >= 0 corresponds to column i, index i < 0 to row -i-1"
                    VarBasisInfo = []
                    for idx in BasisInfo:
                        if idx >= 0 : VarBasisInfo.append(idx)
                    VarBasisInfo.sort()

                    print("Basis Info", BasisInfo, VarBasisInfo)

                    # Getting B inverse * A matrix. (This can be approximated using numerical methods instead)
                    current_lp_rows = self.model.getNLPRows() # Retrieving current number of rows in LP.
                    BInv = []
                    for row_num in range(current_lp_rows):
                        BInv.append( self.model.getLPBInvRow(row_num) ) # Gives row of B^(-1)*A matrix for basis matrix B.
                    BInv = np.array(BInv)

                    # # Checking Error
                    # if BInv.shape[0] != len(VarBasisInfo):
                    #     print("Basis solution error", BInv.shape)
                    #     print("\nBasis Indices = ", VarBasisInfo)

                    # Add cut here
                    ICObject = Intersection_Cuts( problem_data, X_sol, Y_sol, Optimal_Y, VarBasisInfo, BInv)
                    ICVector = ICObject.Cut()
                    self.model.addCons( quicksum( ICVector[idx]*self.X[f'{idx}'] for idx in range(XDim)) + quicksum( ICVector[XDim+idx]*self.Y[f'{idx}'] for idx in range(YDim)) >= 1 , name='LLC' )
                    cutsadded = True

            return cutsadded

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
        if self.addcut(True, sol = solution):
            # print("Infeasible")
            return {"result": SCIP_RESULT.INFEASIBLE}
        else:
            # print("Feasible")
            return {"result": SCIP_RESULT.FEASIBLE}

    # def consenfolp(self, constraints, nusefulconss, solinfeasible):
    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        # print("\n Adding Cuts.......")
        if self.addcut(False, sol = None): # sol = solution
            return {"result": SCIP_RESULT.CONSADDED}
            # print("Consadded")
        else:
            return {"result": SCIP_RESULT.FEASIBLE}
            # print("Feasible declared in folp")

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):

        XDim, YDim = self.data[1], self.data[2]
        for idx in range(XDim):
            self.model.addVarLocks(self.X[f'{idx}'], nlocksneg+nlockspos , nlockspos+nlocksneg)
        for idx in range(YDim):
            self.model.addVarLocks(self.Y[f'{idx}'], nlocksneg+nlockspos , nlockspos+nlocksneg)