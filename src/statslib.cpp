#include "statslib.h"

//Function declarations
double newInvTransformHelp(double val, int key, float* paramsArr){
  double returnVal;
  returnVal = stats::qpois( val, paramsArr[0]);
  return returnVal;
}


