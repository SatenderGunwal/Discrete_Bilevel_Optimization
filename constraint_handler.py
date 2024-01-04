from pyscipopt import Model, Conshdlr, quicksum, multidict, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING
import numpy as np
from intersection_cuts import Intersection_Cut

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
            Dy = problem_data[-2]

            # Getting the solutions at current node
            X_sol, Y_sol = [], []
            for idx in range(XDim):
                xt   = self.model.getSolVal(sol, self.X[f'{idx}'])
                X_sol.append(xt)

            for idx in range(YDim):
                yt = self.model.getSolVal(sol, self.Y[f"{idx}"])
                Y_sol.append(yt)

            def Updated_Problem_Data():
                # Get non-zero variable indexes.
                full_solution = X_sol + Y_sol
                print("\nIncumbent Solution = ", full_solution )
                non_zero_vars = [idx for idx,val in enumerate(full_solution) if val > 0]
                # Get Basis Indices for variables and constraints.
                all_basis_indices  = self.model.getLPBasisInd() # Includes both constraints and variables basis info
                print("\nAll Basis Indices Returned By SCIP = ", all_basis_indices )
                basic_variable_idx = [val for val in all_basis_indices if val >= 0]
                ## Getting non-basic rows(i.e. active rows)
                all_model_rows = self.model.getLPRowsData()
                print("\nNUM of getLPRowsData & getNConss = ", len(all_model_rows), len(self.model.getConss()))
                rows_needed = []
                for row in all_model_rows:
                    if row.getBasisStatus() != 'basic':
                        rows_needed.append(row)
                print("\nRows Needed from Original A :: ", len(rows_needed))
                print("\nNon-Basic(Active) Rows = ", [r.getCols() for r in rows_needed])
                # For non-zero variables x_i not in basic, take x_i = 1 as the constraint
                A_hat, b_hat = [], []
                for idx in non_zero_vars:
                    dummy_zero = list(np.zeros(XDim+YDim))
                    if idx not in basic_variable_idx:
                        dummy_zero[idx] = 1
                        A_hat.append(dummy_zero)
                        b_hat.append(1)
                print("\nA_hat,b_hat for fixed variables = ", A_hat, b_hat)
                # Add non-basic constraints to A_hat and b_hat
                ## Sanity Check 1
                if len(rows_needed) != len(basic_variable_idx):
                    print("\n\nNum of Scip Basic Vars not equals Non-Basic(Active) Constraints")
                ## 
                for row in rows_needed:
                    non_zero_coeff    = row.getVals()
                    # print(non_zero_coeff)
                    non_zero_col_obj  = row.getCols()
                    col_names         = [ str(col.getVar()) for col in non_zero_col_obj ]

                    ROW_LHS, ROW_RHS = [], row.getRhs()
                    for xidx in range(XDim):
                        VName = f"t_X{xidx}"
                        try:
                            idx_ = col_names.index(VName)
                            ROW_LHS.append(non_zero_coeff[idx_])
                        except:
                            ROW_LHS.append(0)
                    for yidx in range(YDim):
                        VName = f"t_Y{yidx}"
                        try:
                            idx_ = col_names.index(VName)
                            ROW_LHS.append(non_zero_coeff[idx_])
                        except:
                            ROW_LHS.append(0)

                    A_hat.append(ROW_LHS)
                    b_hat.append(ROW_RHS)

                A_hat = np.array(A_hat)
                b_hat = np.array(b_hat)

                print("\nA_hat,b_hat including Non-Basic Rows= ", A_hat, b_hat)

                # Getting full basic variables
                ## Sanity check 2
                for idx in basic_variable_idx:
                    if full_solution[idx] == 0:
                        print("\n\nGot Basic Solution with Zero Value\n\n")
                ##
                basic_variable_idx += non_zero_vars
                basic_variable_idx = list(set(basic_variable_idx))
                basic_variable_idx.sort()
                print("\nSorted basic_variable_idx = ", basic_variable_idx)
                # print("\nBasic Vars = ", basic_variable_idx)
                B_hat = A_hat[:,basic_variable_idx]

                return A_hat, b_hat, B_hat, basic_variable_idx

            # Getting Optimal Y for current X solution.
            Optimal_Y, FMILP_Objective = self.FMILP( X_sol )
            ConsCheck = Dy @ np.array(Y_sol) - FMILP_Objective

            # Adding Cut
            cutsadded  = False
            if ConsCheck > 0: # => Bilevel Infeasible Solution
                if checkonly:
                    return True # Will cut be added for this solution
                else:
                    print("\n\nImplementing the LazyCut.....\n")
                    # Manually Getting Basis Indexes
                    A_hat, b_hat, B_hat, VarBasisInfo = Updated_Problem_Data()
                    # Add cut here
                    ICObject = Intersection_Cut( A_hat, b_hat, B_hat, problem_data, X_sol, Y_sol, Optimal_Y, VarBasisInfo)
                    ICVector = ICObject.Cut()
                   
                    self.model.addCons( quicksum( -1*ICVector[idx]*self.X[f'{idx}'] for idx in range(XDim)) + quicksum( -1*ICVector[XDim+idx]*self.Y[f'{idx}'] for idx in range(YDim)) <= -1 , name='LLC' )
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