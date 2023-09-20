#!/usr/bin/env pythonsh
import sys, pdb
from time import time
import numpy
from AutoDockFR.utils import saveAutoDockFRPrediction

t0 =  time()                        # Stores the time (float) at the start of the program

if len(sys.argv)==1:  ## print help msg if no input is given
    sys.argv.append('-help')

from AutoDockFR.Docking import AutoDockFR
from AutoDockFR.Param import Params

i = sys.argv.index('-sol')
solFile = sys.argv[i+1]
sys.argv.remove(solFile)
sys.argv.remove('-sol')

input=Params(args=sys.argv[1:])

adfr=AutoDockFR(input)

from MolKit import Read
solution = Read(solFile)

ln = 3
genes = []
lines = solution.parser[0].allLines
while lines[ln][8]=='[':
    genes.extend( [float(x) for x in lines[ln][9:lines[ln].index(']')].split(',')] )
    ln +=1
score = float(lines[ln].split()[-1])

ga = adfr.docking.search
settings = adfr.setting

def mkGenome(startingGenes, score=None):
    individual = ga.pop.model_genome.clone()
    individual.initialize(settings)
    for i,v in enumerate(startingGenes):
        individual[i].set_value(v)
    sc = -individual.score()
    if score:
        if abs(sc-score)>0.01: print 'gene_score=%f expected_score=%f diff=%f'%(sc, score, abs(sc-score))
    return individual

ind = mkGenome(genes, score=score)

#GAminimize = {'nbSteps':int(5*(0.3/GA_localsearchrate)), 'noImproveStop':2, 'max_steps':200, 'MAX_FAIL':6, 'MIN_VAR':0.01}
GAminimize = {'max_steps':300, 'MAX_FAIL':4, 'MIN_VAR':0.01, 'search_rate':0.057}
newInd = ga.minimize(ind, nbSteps=2, noImproveStop=1, **GAminimize)
#newInd = ga.minimize(ind, nbSteps=5, noImproveStop=3, **GAminimize)

print 'starting at ', -ind._fitness_score
print 'minimized to ', -newInd._fitness_score

from AutoDockFR.FTGA import GAFTMotion_BoxTranslation, \
     GAFTMotion_RotationAboutPointQuat, \
     GAFTMotion_RotationAboutAxis, GAFTMotion_SoftRotamer

nbRounds = 10
#for i in range(nbRounds):
from random import random, gauss
best = newInd.clone()
bondsDev = [
    [0.5],
    [1./3., 2./3.],
    [0.25],
    [1./3., 2./3.],
    [1./3., 2./3.],
    [1./3., 2./3.],
    [1./3., 2./3.],
    [0.25, 0.5, 0.75],
    ]
bestFlipped = False
nr = 0
from time import time
t0 = time()
while 1:
    print 'round', nr, 
    newInd1 = newInd.clone()
    offset = 0
    nbtor = 0
    flipped = False
    for motion in newInd.motionObjs:
        nbg = motion.nbGenes
        if motion.active:
            #if random() < 0.5:
            if isinstance(motion, GAFTMotion_SoftRotamer):
                #continue
                #largeVar = [0.333]
                search_rate= 1./18.
                ct1, devs = motion.jitter(ind[offset:offset+nbg], search_rate)#, largeVar*nbg)
                #continue
            elif isinstance(motion, GAFTMotion_BoxTranslation):
                largeVar = [0.01]
                search_rate = 1./3.#1.0
                ct1, devs = motion.jitter(ind[offset:offset+nbg], search_rate, largeVar*nbg)
            elif isinstance(motion, GAFTMotion_RotationAboutPointQuat):
                largeVar = [0.01]
                search_rate = 1./4. #1.0
                ct1, devs = motion.jitter(ind[offset:offset+nbg], search_rate, largeVar*nbg)
            elif isinstance(motion, GAFTMotion_RotationAboutAxis):
                largeVar = [0.333]
                search_rate= 1./8.
                ct1, devs = motion.jitter(ind[offset:offset+nbg], search_rate, largeVar*nbg)
                v = ind[offset]._value
                if nbtor==1:
                    #print ' ***TOR1', v, devs[0], (v+devs[0])*360, abs((v+devs[0])*360-57)
                    if abs((v+devs[0])*360-57)<50.0:
                        flipped = True
                        print 'asa', (v+devs[0])*360., flipped, 
                ## ct1 = 0
                ## if random() < 2./8.: # 1 side torsion is modified
                ##     ct1 = 1
                ##     index = int(random()*len(bondsDev[nbtor]))
                ##     dev = [gauss(bondsDev[nbtor][index],0.1)]
                ##     if v+dev[0]>1.0: dev[0] = dev[0]-1.0
                ##     if nbtor==1:
                ##         if abs((v+dev[0])*360-57)<50.0:
                ##             flipped = True
                ##         print 'asa', (v+dev[0])*360., index, flipped

                nbtor += 1
            i = 0
            for v,d in zip(ind[offset:offset+nbg].values(), devs):
                newInd1[offset+i]._value = v+d
                i += 1
        offset += nbg    

    print -newInd1.score(),
    newInd2 = ga.minimize(newInd1, nbSteps=10, noImproveStop=3, **GAminimize)
    if newInd2._fitness_score > best._fitness_score:
        best = newInd2.clone()
        bestFlipped = flipped
        ga.saveIndividualPDBQT(best, 'bestLig.pdbqt', 0, comments=[], recFilename='bestRec.pdbqt')
    ga.saveIndividualPDBQT(best, 'ind%d.pdbqt'%nr, 0, comments=[], recFilename='bestRec.pdbqt')
    print '-->', -newInd2.score(), -best._fitness_score, bestFlipped, time()-t0, newInd2.scorer.numEval
    newInd = newInd2
    nr += 1
raise

GAminimize = {'max_steps':300, 'MAX_FAIL':6, 'MIN_VAR':0.01, 'search_rate':0.114}
newInd2 = ga.minimize(newInd, nbSteps=10, noImproveStop=3, **GAminimize)
raise






# create a gene an set it to the starting values
ga = adfr.docking.search
settings = adfr.setting

def mkGenome(startingGenes, score=None):
    individual = ga.pop.model_genome.clone()
    individual.initialize(settings)
    for i,v in enumerate(startingGenes):
        individual[i].set_value(v)
    sc = -individual.evaluate()
    if score:
        if abs(sc-score)>0.01: print 'gene_score=%f expected_score=%f diff=%f'%(sc, score, abs(sc-score))
    return individual

startingGenes =  [1., 0., 0., 0.,   0.5, 0.5, 0.5] + [0.0]*(len(ga.pop[0])-7)

# 9HVP_9HVP -21.447687  rmsd:   0.18
#startingGenes = [0.6138077420873069, 0.0, 0.3091989842856502, 0.4440070278535581, 0.5086692164616329, 0.09990713620402328, 0.44442723584549876, 0.5226121004089779, 0.1069754277837255, 0.7763045230789748, 0.033967373202008724, 0.0007709461418598993, 0.9745524716879868, 0.006043681033362299]

# 9HVP_9HVP -20.162342  rmsd:  12.13
#startingGenes = [0.6768450711277009, 0.5883771545197714, 0.627102370871643, 0.0, 0.4010203847505043, 0.07322940473992212, 0.3667757981895613, 0.40796600705379704, 0.0696800872925774, 0.0, 1.0, 0.09418149345871706, 0.498785533143338, 0.5820756875536732]

# 1tni rmsd:  3.8 E:    -7.499 -> -8.707 50 rounds   437s
startingGenes = [0.2708766635826621, 0.844871580154249, 0.6288376386732152, 0.6687841999875661, 0.4840180701115133, 0.4928854843443466, 0.7162029666631905, 0.3850443390343041, 0.0735301904644645, 0.9585644033564323, 0.7317580658588656, 0.6795038359168699]

# 1tni -8.002780  -> -8.65 10 rounds 10 rounds 150 sec
#startingGenes = [0.030507639525308966, 0.4126899330239841, 0.8341326199519684, 0.3465393452601416, 0.6015626975627226, 0.3902903353303907, 0.4342554611670891, 0.0, 0.9826660983684561, 0.8524259950957274, 0.0015719282993851464, 1.0]

# 1tni -7.108744  -> -7.84 20 round 305s 
#startingGenes = [0.09616782830694151, 0.08755597210429061, 0.5503410001225155, 0.18827474180814632, 0.5068626381599478, 0.43554401252622643, 0.7908786657321147, 0.9702223890380726, 0.16410327178609954, 0.9025539364669721, 0.021665467245311112, 0.8298454554928089]

ind = mkGenome(startingGenes, None)

print ind._score

drot = 0.002
dtrans1 = .012
dtrans1 = dtrans2 = dtrans3 = .008
dtor = .36
absVar = [drot/2., drot/2., drot/2., drot/2.,     # 
          dtrans1/12., dtrans2/8., dtrans3/8]     # 0.012, 0.08 0.08 translation with prob 1 sigma
          #0.36/360, 0.36/360, 0.36/360, 0.36/360, 0.36/360 ]  # 3.6 degrees on torsion with prob 1 sigma
absVar += [dtor/360.]*(len(ind)-7)  # 3.6 degrees on torsion with prob 1 sigma
#absVar = None

print 'absVar', absVar

solisWets = ga.createSolisWets()
final = ga.anneal(ind, solisWets)
print final._score

raise


torsChains = [ [7,8], [9,10,11] ]
def jitter(ind, dtx, dty, dtz):
    from random import uniform

    # rotation
    #rdx = 0.005
    #ind[0]._value += uniform(-rdx, rdx)
    #ind[1]._value += uniform(-rdx, rdx)
    #ind[2]._value += uniform(-rdx, rdx)
    #ind[3]._value += uniform(-rdx, rdx)

    # translation
    ind[4]._value += uniform(-dtx, dtx)
    ind[5]._value += uniform(-dty, dty)
    ind[6]._value += uniform(-dtz, dtz)

    # conformation
    #adx = 0.33 # +/- 120 degrees i.e. 120./360
    adx = 0.16 # +/- 60 degrees i.e. 60./360
    #adx = 0.08 # +/- 30 degrees i.e. 30./360

    for i in range(7,len(ind)):
        mini, maxi = ind[i].bounds
        deltaAngle = uniform(-adx, adx)
        print deltaAngle,
        tmp = ind[i]._value + deltaAngle
        if tmp > maxi:
            tmp = mini + (tmp - maxi)
        elif tmp < mini:
            tmp = maxi + (tmp - mini)
        ind[i]._value = tmp
    print
    
    ## for tc in torsChains:
    ##     if len(tc)==1:
    ##         ind[i]._value += uniform(-adx, adx)
    ##     elif len(tc)==2:
    ##         i1, i2 = tc
    ##     else:
    ##         tcc = tc[:]
    ##         i1 = int(uniform(0, len(tcc)))
    ##         i1 = tcc.pop(i1)
    ##         i2 = int(uniform(0, len(tcc)))
    ##         i2 = tcc[i2]
    ##     deltaAngle = uniform(-adx, adx)
    ##     print i1, i2, deltaAngle
    ##     ind[i1]._value += deltaAngle
    ##     ind[i2]._value -= deltaAngle
            
    ind.evaluate(force=1)

t0 = time()
winner = ind.clone()
mini = ind.clone()
rfail = 0
dx= 1.0
dxMult = 0
fc = 0
nbRounds = 10
roundFails = 3
nbRounds = 50
roundFails = 10
save = False

from AutoDockFR.utils import saveAutoDockFRPrediction

# iterate the process n times
for i in range(nbRounds):
    print "round: %d ene:%f in %.2f(s) dx:%f"%(i, -winner._score, time()-t0, dx)
    fail = 0 # count the number of times SolisWets does not improve
    maxFail = len(ind) # maximum number of such failures
    ct = 0  # counter 
    if save:
        saveAutoDockFRPrediction(
            mini, adfr.docking.setting, adfr.docking.scoreObject,
            R_tree = adfr.docking.ReceptorTree,
            L_tree = adfr.docking.LigandTree, 
            recName=None, ligName ="mini_r%03d_0000.pdb"%i)
    while fail<maxFail: # as long as we do not fail nbGenes time in a row
        # do one local search with lots of steps and allwo to fail 2*nbGenes
        first = True
        new, nbSteps = solisWets.search(mini, max_steps=1000, MAX_FAIL=maxFail*2, MIN_VAR=0.001, absMinVar=absVar)
        # if the results is better it will be minimized again and reset fail counter
        if first and new._score < winner._score*0.98:
            break
        if new._score > mini._score:
            print '  ',fail, nbSteps, mini._score, new._score, new._score- mini._score
            fail = 0
            mini = new
        else: # increment failure count
            fail += 1
        if save:
            saveAutoDockFRPrediction(
                mini, adfr.docking.setting, adfr.docking.scoreObject,
                R_tree = adfr.docking.ReceptorTree,
                L_tree = adfr.docking.LigandTree, 
                recName=None, ligName ="mini_r%03d_%04d.pdb"%(i, ct))
        ct += 1
        # if we reach 10 iterations and score is not better we stop minimizing
        #if ct > 10 and new._score < winner._score: break'
        if ct > 10: break

    # if the minimized ind is better he becomes the winner
    if mini._score > winner._score:
        winner = mini.clone()
        rfail = 0
        ## if dxMult > 0:
        ##     dx -= 1.0 # if dx was increased, decrease it
        ##     dxMult -= 1
    else:
        # we start from the last winner
        mini = winner.clone()
        rfail += 1
        if rfail==roundFails: # if the winner did not improve in 5 round increase dx
            break
            ## dx += 1.0
            ## dxMult += 1
            ## rfail = 0

    # we randomly move the individual using a Gaussian with dev dx
    jitter(mini, dx/12., dx/8., dx/8.)

print 'AFTER search', winner.evaluate(force=1), time()-t0

new, nbSteps = solisWets.search(winner, max_steps=1000, MAX_FAIL=maxFail*2, MIN_VAR=0.0005)

print 'AFTER final mini', new.evaluate(force=1)

adfr.docking.scoreObject.printAllScoreTerms()

# compute final RMSD
from mglutil.math.rmsd import RMSDCalculator
from AutoDockFR.utils import orderRefMolAtoms

for refMolName in adfr.setting['rmsdRef']:
    refMol = Read(refMolName)[0]
    refAts=refMol.getAtoms()

    ligAts = adfr.docking.ligandSet
    # Make sure the ligAts match the order of the reference
    sortedRefAts = orderRefMolAtoms(refAts, ligAts)
    rmsdCalc = RMSDCalculator(refCoords=sortedRefAts.coords)

    a, b, lcoords = adfr.docking.gnm.toPhenotype(new)
    rmsd = rmsdCalc.computeRMSD(lcoords)
    print 'RMSD: %f %s'%(rmsd, refMolName)

if save:
    saveAutoDockFRPrediction(
        new, adfr.docking.setting, adfr.docking.scoreObject,
        R_tree = adfr.docking.ReceptorTree,
        L_tree = adfr.docking.LigandTree, 
        recName="rec.pdb", ligName ="winner1.pdb")

