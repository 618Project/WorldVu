#include "SystemUtil.h"

#ifndef __CHECKPOINT_H__
#define __CHECKPOINT_H__

static double start=0, end=0;
void time_checkpoint(std::string token) {
  // static std::clock_t start=0, end=0;
  std::thread::id this_id = std::this_thread::get_id();
  
  if (start != 0) {
    // end = std::clock();
    end = surround360::util::getCurrTimeSec();
    std::cout << "[" << this_id << "] Time for " << token << " is: " << \
        (end-start) << std::endl;
    // start = std::clock();
    start = surround360::util::getCurrTimeSec();
  } else {
    // start = std::clock();
    start = surround360::util::getCurrTimeSec();
  }

}

#endif
