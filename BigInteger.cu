#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cmath>
#include <ctime>

class BigInteger {
public: 
	enum Sign { negative = -1, zero = 0, positive = 1 };

	Sign sign;
	unsigned int totalBlockCount;
	unsigned int * blocks;

public:
#pragma region Utilities
	__device__ __host__ BigInteger * sqrt() {
		BigInteger * a = BigInteger::toValue((unsigned __int64)1);
		BigInteger * b = MakeACopy();
		*b >>= 5;
 		*b += 8;

		int loopCount = 0;
		while(*b >= *a) {
			BigInteger* mid = a->MakeACopy();
			*mid += *b;
			*mid >>= 1;

			BigInteger* midSqr = mid->MakeACopy();
			*midSqr *= *mid;
			if(*midSqr > *this) {
				*mid -= 1;
				*b = *mid;
			}
			else {
				*mid += 1;
				*a = *mid;
			}

			free(mid->blocks);
			free(mid);
			free(midSqr->blocks);
			free(midSqr);
			loopCount++;
		}

		*a -= 1;
		*this = *a;

		free(a->blocks);
		free(a);
		free(b->blocks);
		free(b);

		return this;
	}

	__device__ __host__ void toString(char * buffer) {
		if(sign == zero) {
			*(buffer++) = '0';
			*buffer = 0;
		}
		else {
			bool is_negative = sign == negative;
			if(is_negative) {
				-(*this);
				*(buffer++) = '-';
			}

			BigInteger * a = toValue((unsigned __int64)1, totalBlockCount);

			while(true) {
				*a *= 10;

				if(*a > *this) {
					*a /= 10;
					break;
				}
			}
		
			BigInteger * b = MakeACopy();
			while(*a >= 10) {
				BigInteger * c = a->MakeACopy();
				BigInteger * d = b->MakeACopy();

				*d /= *c;
				*b = *c;
				free(c->blocks);
				free(c);
				*a /= 10;

				*(buffer++) = '0' + d->blocks[0];
				free(d->blocks);
				free(d);
			}
		
			free(a->blocks);
			free(a);
			*(buffer++) = '0' + b->blocks[0];
			free(b->blocks);
			free(b);
			*buffer = 0;
			if(is_negative) -(*this);
		}
	}

	__device__ __host__ void IsZeroUpdate() {
		bool isZero = true;
		for(int i = 0; i < totalBlockCount; i++) {
			if(blocks[i] != 0) { isZero = false; break; }
		}
		if(isZero) sign = zero;
	}

	__device__ __host__ BigInteger& operator <<=(unsigned int x) {
		if(sign == zero || x == 0) { }
		else {
			if(x/32 > 0) {
				IncreaseIntegerSize(x / 32);

				for(int i = totalBlockCount - 1; i >= (x / 32); i--)
					blocks[i] = blocks[i-(x/32)];
			}
			unsigned int carry = 0;
			for(int i = 0; i < (x / 32); i++) blocks[i] = 0;
			for(int i = 0; i < totalBlockCount; i++) {
				unsigned __int64 thisBlock = (unsigned __int64)blocks[i] << (x % 32);
				blocks[i] = (unsigned int)thisBlock | carry;
				carry = (unsigned int)(thisBlock >> 32);
			}

			if(carry) {
				IncreaseIntegerSize(1);
				blocks[totalBlockCount - 1] = carry;
			}
		}

		return *this;
	}

	__device__ __host__ BigInteger& operator >>=(unsigned int x) {
		if(sign == zero || x == 0) { }
		else if(x >= 32*(totalBlockCount-1))
			*this = 0;
		else {
			int i = 0;
			for(; i < totalBlockCount - (x/32); i++)
				blocks[i] = blocks[i+(x/32)];
			unsigned int carry = 0;
			for(; i < totalBlockCount; i++) blocks[i] = 0;
			for(int i = totalBlockCount-1; i >= 0; i--) {
				unsigned __int64 thisBlock = ((unsigned __int64)blocks[i] << 32) >> (x % 32);
				blocks[i] = (unsigned int)(thisBlock >> 32) | carry;
				carry = (unsigned int)thisBlock;
			}
		}

		return *this;
	}

	__device__ __host__ void operator -() {
		if(sign == negative) sign = positive;
		else if(sign == positive) sign = negative;
	}

	__device__ __host__ void ResizeInteger(unsigned int newBlockCount) {
		unsigned int * old_blocks = blocks;
		blocks = (unsigned int*)malloc(newBlockCount*sizeof(unsigned int));
		memset(blocks, 0, newBlockCount*sizeof(unsigned int));
		memcpy(blocks, old_blocks, ((newBlockCount > totalBlockCount) ? totalBlockCount : newBlockCount) * sizeof(unsigned int));
		
		totalBlockCount = newBlockCount;
		free(old_blocks);
	}
	
	__device__ __host__ void SetBitAtOffset(unsigned int offset) {
		if(totalBlockCount < ((offset+1) / 32) + (((offset+1) % 32) ? 1 : 0))
			ResizeInteger(((offset+1) / 32) + (((offset+1) % 32) ? 1 : 0));

		if(sign == zero) sign = positive;

		blocks[offset / 32] |= (1 << (offset % 32));
	}

	__device__ __host__ void IncreaseIntegerSize(unsigned int additional_block_count) {
		ResizeInteger(totalBlockCount + additional_block_count);
	}

	__device__ __host__ BigInteger* MakeACopy() {
		BigInteger*out = (BigInteger*)malloc(sizeof(BigInteger));
		out->blocks = (unsigned int*)malloc(totalBlockCount * sizeof(unsigned int));
		out->totalBlockCount = totalBlockCount;
		out->sign = sign;
		memset(out->blocks, 0, totalBlockCount);
		memcpy(out->blocks, blocks, totalBlockCount * sizeof(unsigned int));
		return out;
	}

	__device__ __host__ static BigInteger* toValue(char * value, unsigned int block_count) {
		BigInteger * out = toValue((unsigned __int64)0, block_count);
		bool outNeg = (*value == '-');
		if(outNeg) value++;
		unsigned int digitCount = strlen(value);
		BigInteger* a = toValue((unsigned __int64)1, block_count);
		for(int i = 0; i < digitCount - 1; i++)
			*a *= 10;
		while(*a > 1) {
			BigInteger * b = a->MakeACopy();
			*b *= (*(value++) - '0');
			*out += *b;
			free(b->blocks);
			free(b);
			*a /= 10;
		}
		*a *= (*value - '0');
		*out += *a;
		free(a->blocks);
		free(a);

		if(outNeg) out->sign = negative;
		out->IsZeroUpdate();
		return out;
	}

	__device__ __host__ static BigInteger* toValue(Sign sign, unsigned int * digits, unsigned int block_count) {
		BigInteger * out = (BigInteger*)malloc(sizeof(BigInteger));
		out->blocks = (unsigned int*)malloc(block_count*sizeof(unsigned int));
		memcpy(out->blocks, digits, block_count*sizeof(unsigned int));
		out->totalBlockCount = block_count;
		out->sign = sign;
		return out;
	}

	__device__ __host__ static BigInteger* toValue(__int64 value, unsigned int block_count) {
		if(block_count < 2) block_count = 2;

		BigInteger * out = (BigInteger*)malloc(sizeof(BigInteger));
		out->blocks = (unsigned int*)malloc(block_count*sizeof(unsigned int));
		memset(out->blocks, 0, block_count*sizeof(unsigned int));
		out->sign = positive;
		if(value == 0) 
			out->sign = zero;
		else if(value < 0) {
			out->sign = negative;
			value = -value;
		}
		
		out->totalBlockCount = block_count;
		out->blocks[0] = value & 0xFFFFFFFF;
		out->blocks[1] = (value >> 32) & 0xFFFFFFFF;
		return out;
	}

	__device__ __host__ static BigInteger* toValue(__int64 value) {
		return toValue(value, 2);
	}
	
	__device__ __host__ static BigInteger* toValue(unsigned __int64 value, unsigned int block_count) {
		if(block_count < 2) block_count = 2;

		BigInteger* out = new BigInteger;
		out->sign = positive;
		if(value == 0) 
			out->sign = zero;
		out->blocks = new unsigned int[block_count];
		memset(out->blocks, 0, block_count*sizeof(unsigned int));

		out->totalBlockCount = block_count;
		out->blocks[0] = value & 0xFFFFFFFF;
		out->blocks[1] = (value >> 32) & 0xFFFFFFFF;
		return out;
	}

	__device__ __host__ static BigInteger* toValue(unsigned __int64 value) {
		return toValue(value, 2);
	}

	__device__ __host__ static BigInteger* toValue(int value, unsigned int block_count) {
		if(block_count < 1) block_count = 1;
		
		BigInteger *out = (BigInteger*)malloc(sizeof(BigInteger));
		out->blocks = (unsigned int*)malloc(block_count*sizeof(unsigned int));
		memset(out->blocks, 0, block_count*sizeof(unsigned int));
		out->sign = positive;
		if(value == 0) 
			out->sign = zero;
		else if(value < 0) {
			out->sign = negative;
			value = -value;
		}
		
		out->totalBlockCount = block_count;
		out->blocks[0] = value;
		return out;
	}

	__device__ __host__ static BigInteger* toValue(int value) {
		return toValue(value, 1);
	}

	__device__ __host__ static BigInteger* toValue(short value, unsigned int block_count) {
		if(block_count < 1) block_count = 1;
		
		BigInteger* out = (BigInteger*)malloc(sizeof(BigInteger));
		out->blocks = (unsigned int*)malloc(block_count*sizeof(unsigned int));
		memset(out->blocks, 0, block_count*sizeof(unsigned int));
		out->sign = positive;
		if(value == 0) 
			out->sign = zero;
		else if(value < 0) {
			out->sign = negative;
			value = -value;
		}
		
		out->totalBlockCount = block_count;
		out->blocks[0] = value;
		return out;
	}

	__device__ __host__ static BigInteger* toValue(short value) {
		return toValue(value, 1);
	}

	__device__ __host__ char toByte() {
		return (char)blocks[0];
	}

	__device__ __host__ unsigned short toUShort() {
		return (unsigned short)blocks[0];
	}

	__device__ __host__ unsigned int toUInt() {
		return blocks[0];
	}
#pragma endregion

#pragma region BigInteger input operations
	__device__ __host__ BigInteger& operator /=(BigInteger &x) {
		Sign thisSign = sign, xSign = x.sign;

		if(thisSign == zero) { } //0 / x = 0 with r of x
		else if(xSign == zero) { //x/0 = undefined. Define it, = 0 with r of this
			x = *this;
			*this = 0;
		}
		else {
			if(thisSign == negative) -(*this);
			if(xSign	== negative) -x;
		
			if(*this < x) {
				x = *this;
				*this = 0;
			}
			else {
				unsigned int shiftCount = 0;
				BigInteger * returnVal = BigInteger::toValue((unsigned __int64)0);
				bool reAdjust = false;
				while(*this > x) {
					reAdjust = true;
					x <<= 1;
					shiftCount++;
				}
			
				if(reAdjust) {
					x >>= 1;
					shiftCount--;
				}

				for(int i = shiftCount; i >= 0; i--) {
					if(*this >= x) {
						*this -= x;
						returnVal->SetBitAtOffset(i);
					}

					x >>= 1;
				}
				BigInteger * returnRemainder = MakeACopy();
				*this = *returnVal;
				free(returnVal->blocks);
				free(returnVal);
				x = *returnRemainder;
				free(returnRemainder->blocks);
				free(returnRemainder);
			}
		}

		return *this;
	}
	
	__device__ __host__ BigInteger& operator *=(BigInteger &x) {
		if(sign == zero) { } //0*x=0
		else if(x.sign == zero) {
			for(int i = 0; i < totalBlockCount; i++) {
				blocks[i] = 0;
			}
		}
		else {
			Sign final_sign = (sign != x.sign) ? negative : positive;

			BigInteger * running_total = BigInteger::toValue((unsigned __int64)0, totalBlockCount);
			for(int i = 0; i < totalBlockCount; i++)
				for(int j = 0; j < x.totalBlockCount; j++) {
					BigInteger * amountToAdd = BigInteger::toValue(((unsigned __int64)blocks[i] * (unsigned __int64)x.blocks[j]));
					*amountToAdd <<= ((i + j) * 32);
					*running_total += *amountToAdd;
					free(amountToAdd->blocks);
					free(amountToAdd);
				}

			running_total->sign = final_sign;

			*this = *running_total;
			free(running_total->blocks);
			free(running_total);
		}

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ BigInteger& operator |=(BigInteger &x) {
		if(totalBlockCount < x.totalBlockCount) ResizeInteger(x.totalBlockCount);
			
		for(int i = 0; i < x.totalBlockCount; i++)
			blocks[i] |= x.blocks[i];

		//sign bit is ored too
		sign = (x.sign == negative || sign == negative) ? negative : positive;

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ BigInteger& operator &=(BigInteger &x) {
		 int i = 0;
		for(; i < x.totalBlockCount && i < totalBlockCount; i++)
			blocks[i] &= x.blocks[i];
		for(; i < totalBlockCount; i++)
			blocks[i] = 0;

		//sign bit is ored too
		sign = (x.sign == negative && sign == negative) ? negative : positive;

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ BigInteger& operator +=(BigInteger &x) {
		if(x.sign == zero) { } //+= 0;, do nothing
		else if(sign == zero) //0 += x;, x
			*this = x;
		else if(sign == x.sign) {
			int k = x.totalBlockCount;
			if(totalBlockCount < x.totalBlockCount) {
				//Get most significant digit
				for(; x.blocks[k - 1] == 0; k--);
				if(k > totalBlockCount) ResizeInteger(k);
			}
			
			unsigned int carryBlock = 0;
			int i = 0;
			for(; i < k; i++) {
				unsigned __int64 thisBlock = (unsigned __int64)carryBlock + (unsigned __int64)blocks[i] + (unsigned __int64)x.blocks[i];
				blocks[i] = (unsigned int)thisBlock;
				carryBlock = (unsigned int)(thisBlock >> 32);				
			}

			if(carryBlock) {
				if(i >= totalBlockCount) { IncreaseIntegerSize(1); }
				blocks[i] += carryBlock;
			}
		}
		else if(sign == negative) {
			BigInteger * buffer = MakeACopy();
			*this = x;
			-(*buffer);
			*this -= *buffer;
			free(buffer->blocks);
			free(buffer);
		}
		else if(x.sign == negative) { //arbatrairy if, this is the only other option
			-x;
			*this -= x;
			-x;
		}

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ BigInteger& operator -=(BigInteger &x) {
		if(x.sign == zero) { } // -= 0, do nothing
		else if(sign == zero) { //0 -= x, -x
			*this = x;
			-(*this);
		}
		else if(sign == x.sign) {
			if((sign == negative && x < *this) || (sign == positive && *this < x)) {
				BigInteger * buffer = MakeACopy();
				*this = x;
				*this -= *buffer;
				-(*this);
				free(buffer->blocks);
				free(buffer);
			}
			else {
				bool hadToCarry = false;
				for(int i = 0; i < x.totalBlockCount; i++) {
					unsigned __int64 thisBlock = blocks[i];
					if(hadToCarry) thisBlock--;
					hadToCarry = (thisBlock < x.blocks[i]);
					if(hadToCarry) thisBlock += 0x100000000;
					blocks[i] = thisBlock - x.blocks[i];
				}
			}
		}
		else if(sign == negative) {
			-(*this);
			*this += x;
			-(*this);
		}
		else if(x.sign == negative) {
			-x;
			*this += x;
			-x;
		}

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ bool operator <(BigInteger &x) {
		bool returnVal = false;
		if(sign == zero) {
			if(x.sign == positive) returnVal = true;
		}
		else if(x.sign == zero) {
			if(sign == negative) returnVal = true;
		}
		else if(x.sign == sign) {
			int i = (totalBlockCount < x.totalBlockCount) ? x.totalBlockCount-1 : totalBlockCount-1;
			for(; i >= totalBlockCount; i--)
				if(x.blocks[i] != 0) {
					returnVal = (sign == negative) ? false : true;
					goto endThis;
				}
			for(; i >= x.totalBlockCount; i--)
				if(blocks[i] != 0) {
					returnVal = (sign == negative) ? true : false;
					goto endThis;
				}
			for(; i >= 0; i--) {
				if(blocks[i] == x.blocks[i]) continue;
				else if(blocks[i] < x.blocks[i]) { returnVal = (sign == negative) ? false : true; break; }
				else { returnVal = (sign == negative) ? true : false; break; }
			}
		}
		else if(x.sign == positive)
			returnVal = true;
		
		endThis:
		return returnVal;
	}

	__device__ __host__ bool operator >(BigInteger &x) {
		bool returnVal = false;
		if(sign == zero) {
			if(x.sign == positive) returnVal = true;
		}
		else if(x.sign == zero) {
			if(sign == negative) returnVal = false;
		}
		else if(x.sign == sign) {
			int i = (totalBlockCount < x.totalBlockCount) ? x.totalBlockCount-1 : totalBlockCount-1;
			for(; i >= totalBlockCount; i--)
				if(x.blocks[i] != 0) {
					returnVal = (sign == negative) ? true : false;
					goto endThis;
				}
			for(; i >= x.totalBlockCount; i--)
				if(blocks[i] != 0) {
					returnVal = (sign == negative) ? false : true;
					goto endThis;
				}
			for(; i >= 0; i--) {
				if(blocks[i] == x.blocks[i]) continue;
				else if(blocks[i] > x.blocks[i]) { returnVal = (sign == negative) ? false : true; break; }
				else { returnVal = (sign == negative) ? true : false; break; }
			}
		}
		else if(x.sign == positive)
			returnVal = true;
		
		endThis:
		return returnVal;
	}

	__device__ __host__ bool operator >=(BigInteger &x) {
		return !(*this < x);
	}

	__device__ __host__ bool operator <=(BigInteger &x) {
		return !(*this > x);
	}

	__device__ __host__ bool operator ==(BigInteger &x) {
		bool returnVal = true;

		if(x.sign != sign) { returnVal = false; goto endThis; }

		int i = 0;
		for(; i < totalBlockCount && i < x.totalBlockCount; i++)
			if(blocks[i] != x.blocks[i]) { returnVal = false; goto endThis; }

		for(; i < x.totalBlockCount; i++)
			if(0 != x.blocks[i]) { returnVal = false; goto endThis; }

		for(; i < totalBlockCount; i++)
			if(0 != blocks[i]) { returnVal = false; goto endThis; }

		endThis:
		return returnVal;
	}
	
	__device__ __host__ BigInteger& operator =(BigInteger &x) {
		if(totalBlockCount < x.totalBlockCount) {
			free(blocks);
			blocks = (unsigned int*)malloc(x.totalBlockCount * sizeof(unsigned int));
			totalBlockCount = x.totalBlockCount;
		}
		sign = x.sign;
		memset(blocks, 0, totalBlockCount* sizeof(unsigned int));
		memcpy(blocks, x.blocks, x.totalBlockCount * sizeof(unsigned int));
		return *this;
	}
#pragma endregion

#pragma region unsigned __int64 operations
	__device__ __host__ BigInteger& operator %=(unsigned __int64 x) {
		BigInteger * deleteThis = toValue(x);
		*this /= *deleteThis;
		*this = *deleteThis;
		free(deleteThis->blocks);
		free(deleteThis);
		return *this;
	}

	__device__ __host__ BigInteger& operator /=(unsigned __int64 x) {
		BigInteger * deleteThis = toValue(x);
		*this /= *deleteThis;
		free(deleteThis->blocks);
		free(deleteThis);
		return *this;
	}

	__device__ __host__ BigInteger& operator *=(unsigned __int64 x) {
		if(sign == zero) { } //0*x=0
		else if(x == 0) {
			for(int i = 0; i < totalBlockCount; i++) {
				blocks[i] = 0;
			}
		}
		else {
			Sign final_sign = (sign == negative) ? negative : positive;

			BigInteger * running_total = BigInteger::toValue(0, totalBlockCount);
			for(int i = 0; i < totalBlockCount; i++)
				for(int j = 0; j < 2; j++) {
					BigInteger * amountToAdd = BigInteger::toValue((unsigned __int64)blocks[i] * (unsigned __int64)((unsigned int)(x >> (j*32))));
					*amountToAdd <<= ((i + j) * 32);
					*running_total += *amountToAdd;
					free(amountToAdd->blocks);
					free(amountToAdd);
				}

			running_total->sign = final_sign;

			*this = *running_total;
			free(running_total->blocks);
			free(running_total);
		}

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ BigInteger& operator |=(unsigned __int64 x) {
		if(totalBlockCount < 2) ResizeInteger(2);
			
		for(int i = 0; i < 2; i++)
			blocks[i] |= (x >> (i*32));

		//sign bit is ored too
		sign = (sign == negative) ? negative : positive;

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ BigInteger& operator &=(unsigned __int64 x) {
		if(totalBlockCount < 2) ResizeInteger(2);
			
		int i = 0;
		for(; i < 2; i++)
			blocks[i] &= (x >> (i*32));
		for(; i < totalBlockCount; i++)
			blocks[i] = 0;

		//sign bit is anded too
		sign = positive;

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ BigInteger& operator +=(unsigned __int64 x) {
		if(sign == zero) { //0 += x;, x
			if(totalBlockCount < 2) ResizeInteger(2);
			blocks[0] = (unsigned int)x;
			blocks[1] = (unsigned int)(x >> 32);
			sign = positive;
		}
		else if(sign == positive) {
			if(totalBlockCount < 2) ResizeInteger(2);
			
			unsigned int carryBlock = 0;
			for(int i = 0; i < 2; i++) {
				unsigned __int64 thisBlock = (unsigned __int64)carryBlock + (unsigned __int64)blocks[i] + (unsigned __int64)(x >> (i*32));
				blocks[i] = (unsigned int)thisBlock;
				carryBlock = (unsigned int)(thisBlock >> 32);				
			}

			if(carryBlock) {
				IncreaseIntegerSize(1);
				blocks[totalBlockCount-1] = carryBlock;
			}
		}
		else if(sign == negative) {
			BigInteger * buffer = MakeACopy();
			if(totalBlockCount < 2) ResizeInteger(2);
			blocks[0] = (unsigned int)x;
			blocks[1] = (unsigned int)(x >> 32);
			-(*buffer);
			*this -= *buffer;
			free(buffer->blocks);
			free(buffer);
		}

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ BigInteger& operator -=(unsigned __int64 x) {
		if(sign == zero) { //0 -= x, -x
			if(totalBlockCount < 2) ResizeInteger(2);
			blocks[0] = (unsigned int)x;
			blocks[1] = (unsigned int)(x >> 32);
			-(*this);
		}
		else if(sign == positive) {
			if(*this < x) {
				BigInteger * buffer = MakeACopy();
				if(totalBlockCount < 2) ResizeInteger(2);
				blocks[0] = (unsigned int)x;
				blocks[1] = (unsigned int)(x >> 32);
				*this -= *buffer;
				-(*this);
				free(buffer->blocks);
				free(buffer);
			}
			else {
				bool hadToCarry = false;
				for(int i = 1; i >= 0; i--) {
					unsigned __int64 thisBlock = blocks[i];
					if(hadToCarry) thisBlock--;
					hadToCarry = (thisBlock < (x >> (i*32)));
					if(hadToCarry) thisBlock += 0x100000000;
					blocks[i] = thisBlock - (x >> (i*32));
				}
			}
		}
		else if(sign == negative) {
			-(*this);
			*this += x;
			-(*this);
		}

		IsZeroUpdate();
		return *this;
	}

	__device__ __host__ bool operator <(unsigned __int64 x) {
		bool returnVal = false;
		if((sign == zero && x != 0) || sign == negative)
			returnVal = true;
		else if(positive == sign) {
			int i = (totalBlockCount < 2) ? 1 : totalBlockCount-1;
			for(; i >= 2; i--)
				if(blocks[i] != 0) {
					returnVal = false;
					goto endThis;
				}
			for(; i >= totalBlockCount; i--)
				if((unsigned int)(x >> (32*i)) != 0) {
					returnVal = true;
					goto endThis;
				}
			for(; i >= 0; i--) {
				if(blocks[i] == (x >> (32*i))) continue;
				else if(blocks[i] < (x >> (32*i))) { returnVal = true; break; }
				else { returnVal = false; break; }
			}
		}
		
		endThis:
		return returnVal;
	}

	__device__ __host__ bool operator >(unsigned __int64 x) {
		bool returnVal = false;
		if((sign == zero && x != 0) || sign == negative) {
			returnVal = false;
		}
		else if(positive == sign) {
			int i = (totalBlockCount < 2) ? 1 : totalBlockCount-1;
			for(; i >= totalBlockCount; i--)
				if((unsigned int)(x >> (32*i)) != 0) {
					returnVal = false;
					goto endThis;
				}
			for(; i >= 2; i--)
				if(blocks[i] != 0) {
					returnVal = true;
					goto endThis;
				}
			for(; i >= 0; i--) {
				if(blocks[i] == (x >> (32*i))) continue;
				else if(blocks[i] > (x >> (32*i))) { returnVal = true; break; }
				else { returnVal = false; break; }
			}
		}
		
		endThis:
		return returnVal;
	}

	__device__ __host__ bool operator >=(unsigned __int64 x) {
		return !(*this < x);
	}

	__device__ __host__ bool operator <=(unsigned __int64 x) {
		return !(*this > x);
	}

	__device__ __host__ bool operator ==(unsigned __int64 x) {
		bool returnVal = true;

		int i = 0;
		for(; i < totalBlockCount && i < 2; i++)
			if(blocks[i] != (x >> (i*32))) { returnVal = false; goto endThis; }

		for(; i < 2; i++)
			if(0 != (x >> (i*32))) { returnVal = false; goto endThis; }

		for(; i < totalBlockCount; i++)
			if(0 != blocks[i]) { returnVal = false; goto endThis; }

		endThis:
		return returnVal;
	}
	
	__device__ __host__ BigInteger& operator =(unsigned __int64 x) {
		if(totalBlockCount < 2) ResizeInteger(2);
		int i = 0;
		blocks[i++] = (unsigned int)x;
		blocks[i++] = (unsigned int)(x >> 32);

		for(; i < totalBlockCount; i++)
			blocks[i] = 0;

		sign = (x == 0) ? zero : positive;
		return *this;
	}
#pragma endregion
};