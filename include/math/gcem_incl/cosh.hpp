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
 * compile-time hyperbolic cosine function
 */

#ifndef _gcem_cosh_HPP
#define _gcem_cosh_HPP

template<typename T>
constexpr
T
cosh_int(const T x)
{
    return( // indistinguishable from zero
            GCLIM<T>::epsilon() > abs(x) ? \
                T(1) : 
            // else
                (exp(x) + exp(-x)) / T(2) );
}

template<typename T>
constexpr
return_t<T>
cosh(const T x)
{
    return cosh_int<return_t<T>>(x);
}

#endif
