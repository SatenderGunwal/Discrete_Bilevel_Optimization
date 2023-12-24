from pyscipopt import Model, Conshdlr, quicksum, multidict, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING
import numpy as np
from intersection_cuts import Intersection_Cuts

class Constraint_Handler(Conshdlr):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def FMILP(self, HPR_Solution_X ):

        problem_data, XDim, YDim, _ = self.data
        [A, B, C, Dy]               = problem_data[-5:-1]

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

            problem_data, XDim, YDim, storage = self.data
            [Gx, Gy, G0, A, B, C, Dy] = problem_data[-8:-1]
            init_NConstraints         = Gx.shape[0] + A.shape[0] # Number of Problem Constraints (Without Cuts)

            # Getting the solutions at current node
            X_sol, Y_sol = [], []
            for idx in range(XDim):
                xt   = self.model.getSolVal(sol, self.X[f'{idx}'])
                X_sol.append(xt)

            for idx in range(YDim):
                yt = self.model.getSolVal(sol, self.Y[f"{idx}"])
                Y_sol.append(yt)

            def getLPBasis():
                    
                NumConstrs_  = self.model.getNConss() #len(constraints_)#self.model.getNLPRows()
                print("\nNConss = ", NumConstrs_)
                Non_zero_idx = []
                Zero_idx     = []
                sol_         = X_sol + Y_sol
                for idx,val in enumerate(sol_):
                    if val > 0:
                        Non_zero_idx.append(idx)
                    else:
                        Zero_idx.append(idx)

                zero_needed = NumConstrs_ - len(Non_zero_idx)

                if zero_needed > 0:
                    BasisIndexes = Non_zero_idx + Zero_idx[:zero_needed]
                else:
                    print("Basic vars more than constraints ....")
                    print("X = ", sol_[:XDim])
                    print("Y = ", sol_[XDim:])
                    BasisIndexes = Non_zero_idx
                BasisIndexes.sort()
                print("\nNum of Basic Vars = ", len(BasisIndexes))
                return BasisIndexes
            
            def getNewData():
                # Setting Basic Variables Indexes
                constraints_      = self.model.getConss() # Get Constraint Objects
                total_constraints = self.model.getNConss() # getDualSolVal(constraint)
                
                ANew_, BNew_, CNew_ = A, B, C

                print("\nInit/total_constraints = ", init_NConstraints, total_constraints)
                if total_constraints > init_NConstraints: # Data is Updated only if New Constraints Are Added

                    for con in constraints_[init_NConstraints:]:
                        constr_coef_dict = self.model.getValsLinear(con) # Only Provides Non-Zero Coefficients (dict values -> t_X or t_Y)
                        coef_dictKeys  = list(constr_coef_dict.keys())
                        print("\nDictionary = ", constr_coef_dict )
                        print("NameVars = ", coef_dictKeys)
                        # print("Coeffs = ", constr_coef_dict, "\n", self.model.getLhs(con), self.model.getRhs(con))

                        # Retrieving Coefficients of this constraint from incomplete dictionary constr_coef_dict
                        XVar_Coeffs, YVar_Coeffs = [], []
                        for xidx in range(XDim):
                            VName = f"t_X{xidx}"
                            if VName in coef_dictKeys:
                                XVar_Coeffs.append(constr_coef_dict[VName])
                            else:
                                XVar_Coeffs.append(0)
                        for yidx in range(YDim):
                            VName = f"t_Y{yidx}"
                            if VName in coef_dictKeys:
                                YVar_Coeffs.append(constr_coef_dict[VName])
                            else:
                                YVar_Coeffs.append(0)

                        # Updating Data
                        ANew_ = np.vstack((ANew_,np.array([XVar_Coeffs])))
                        BNew_ = np.vstack((BNew_,np.array([YVar_Coeffs])))
                        CNew_ = np.array(list(CNew_) + [1])

                return ANew_, BNew_, CNew_

            # Getting Optimal Y for current X solution.
            Optimal_Y, FMILP_Objective = self.FMILP( X_sol )
            ConsCheck = Dy @ np.array(Y_sol) - FMILP_Objective

            # Adding Cut
            cutsadded  = False
            if ConsCheck > 0: # => Bilevel Infeasible Solution
                if checkonly:
                    return True # Will cut be added for this solution
                else:
                    # Manually Getting Basis Indexes
                    VarBasisInfo     = getLPBasis() # function defined above
                    ANew, BNew, CNew = getNewData()
                    print(ANew)
                    print( "\nNew A Shape = ", ANew.shape )
                    print("Basis Info", VarBasisInfo)

                    # Add cut here
                    ICObject = Intersection_Cuts( ANew, BNew, CNew, problem_data, X_sol, Y_sol, Optimal_Y, VarBasisInfo)
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