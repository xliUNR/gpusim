/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
  ##
  ##   This file is part of the GCE-Math C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*
 * compile-time hyperbolic sine function
 */

#ifndef _gcem_sinh_HPP
#define _gcem_sinh_HPP

template<typename T>
constexpr
T
sinh_check(const T x)
{
    return( // indistinguishable from zero
            GCLIM<T>::epsilon() > abs(x) ? \
                T(0) :
            // else
                (exp(x) - exp(-x))/T(2) );
}

template<typename T>
constexpr
return_t<T>
sinh(const T x)
{
    return sinh_check<return_t<T>>(x);
}

#endif
