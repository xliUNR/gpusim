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
 * compile-time power function
 */

#ifndef _gcem_pow_HPP
#define _gcem_pow_HPP

template<typename T>
constexpr
T
pow_dbl(const T base, const T exp_term)
{
    return exp(exp_term*log(base));
}

//

template<typename Ta, typename Tb>
constexpr
Ta
pow(const Ta base, const Tb exp_term)
{
    return( std::is_integral<Tb>::value ? \
            // if
                pow_integral(base,exp_term) :
            // else
                pow_dbl(base,Ta(exp_term)) );
}

#endif
