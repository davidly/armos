/*
    This is a simplistic Arm64/ARMv8 emulator.
    Only physical memory is supported.
    Only Base and SIMD&FP instructions are supported.
    Many instructions (30%?) are not yet implemented.
    I only implemented instructions g++, clang, or Rust compilers emit or use in their runtimes for a handful of test apps.

    Written by David Lee in October 2024

    Useful: https://developer.arm.com/
            https://developer.arm.com/documentation/111182/2025-09_ASL1/SIMD-FP-Instructions/MVNI--Move-inverted-immediate--vector--?lang=en
            https://mariokartwii.com/armv8/ch16.html
*/

#include <stdint.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <bitset>
#include <chrono>

#ifdef _WIN32
#include <intrin.h>
#endif

#include <djl_128.hxx>
#include <djltrace.hxx>

#include "arm64.hxx"

using namespace std;
using namespace std::chrono;

#if defined( __GNUC__ ) && !defined( __APPLE__) && !defined( __clang__ )
#pragma GCC diagnostic ignored "-Wformat="
#endif

// NAN has different values for different compilers
static const uint64_t g_ui64_NAN = 0x7ff8000000000000;
#define MY_NAN ( * (double *) & g_ui64_NAN )

static uint64_t g_State = 0;

const uint64_t stateTraceInstructions = 1;
const uint64_t stateEndEmulation = 2;

bool Arm64::trace_instructions( bool t )
{
    bool prev = ( 0 != ( g_State & stateTraceInstructions ) );
    if ( t )
        g_State |= stateTraceInstructions;
    else
        g_State &= ~stateTraceInstructions;
    return prev;
} //trace_instructions

void Arm64::end_emulation() { g_State |= stateEndEmulation; }

template <class T> T do_abs( T x )
{
    return ( x < 0 ) ? -x : x;
} //do_abs

void Arm64::set_flags_from_nzcv( uint64_t nzcv )
{
    fN = ( 0 != ( nzcv & 8 ) );
    fZ = ( 0 != ( nzcv & 4 ) );
    fC = ( 0 != ( nzcv & 2 ) );
    fV = ( 0 != ( nzcv & 1 ) );
} //set_flags_from_nzcv

uint16_t consider_endian16( uint16_t x )
{
#ifdef TARGET_BIG_ENDIAN
    return flip_endian16( x );
#endif
    return x;
} //consider_endian16

uint32_t consider_endian32( uint32_t x )
{
#ifdef TARGET_BIG_ENDIAN
    return flip_endian32( x );
#endif
    return x;
} //consider_endian32

uint64_t consider_endian64( uint64_t x )
{
#ifdef TARGET_BIG_ENDIAN
    return flip_endian64( x );
#endif
    return x;
} //consider_endian64

static __inline_perf void mcpy( void * d, const void * s, const size_t c ) // memcpy but optimized for small sizes
{
    assert( 1 == c || 2 == c || 4 == c || 8 == c || 16 == c );

    // I tried various permutations and this was fastest for msft c++, clang, and g++. loops fail with the latter two
    if ( 1 == c )
        * ( (uint8_t *) d ) = * ( (uint8_t *) s );
    else if ( 2 == c )
        * ( (uint16_t *) d ) = * ( (uint16_t *) s );
    else if ( 4 == c )
        * ( (uint32_t *) d ) = * ( (uint32_t *) s );
    else
    {
        * ( (uint64_t *) d ) = * ( (uint64_t *) s );
        if ( 16 == c )
            * ( (uint64_t *) d + 1 ) = * ( (uint64_t *) s + 1 );
    }
} //mcpy

static uint64_t count_bits( uint64_t x )
{
#ifdef _M_AMD64
        return __popcnt64( x ); // less portable, but faster. Not on Q9650 CPU and other older Intel CPUs. use code below instead if needed.
#elif defined( __aarch64__ )
        return std::bitset<64>( x ).count();
#else
    uint64_t count = 0;
    while ( 0 != x )
    {
        if ( x & 1 )
            count++;
        x >>= 1;
    }
    return count;
#endif
} //count_bits

static uint32_t count_leading_zeroes32( uint32_t x )
{
#ifdef _WIN32 // if targeting older CPUs you may need to rebuild with the manual code below
    DWORD lz = 0;
    if ( _BitScanReverse( &lz, x ) )
        return 31 - lz;
    return 32;
#elif defined( __GNUC__ ) || defined( __clang__ )
    return __builtin_clz( x );
#else
    uint32_t count = 0;
    while ( x )
    {
        count++;
        x >>= 1;
    }

    return 32 - count;
#endif
} //count_leading_zeroes32

#ifdef _WIN32
__declspec(noinline)
#endif
void Arm64::unhandled()
{
    emulator_hard_termination( *this, "opcode not handled:", op );
} //unhandled

Arm64::FPRounding Arm64::fp_decode_rm( uint64_t rm )
{
    if ( 0 == rm )
        return FPRounding_TIEAWAY;
    if ( 1 == rm )
        return FPRounding_TIEEVEN;
    if ( 2 == rm )
        return FPRounding_POSINF;
    if ( 3 == rm )
        return FPRounding_NEGINF;
    unhandled();
    return FPRounding_TIEEVEN; // keep the compiler happy
} //fp_decode_rm

Arm64::FPRounding Arm64::fp_decode_rmode( uint64_t rmode ) // rmode is what is stored in 23:22 of fpcr
{
    if ( 0 == rmode )
        return FPRounding_TIEEVEN;
    if ( 1 == rmode )
        return FPRounding_POSINF;
    if ( 2 == rmode )
        return FPRounding_NEGINF;
    if ( 3 == rmode )
        return FPRounding_ZERO;
    unhandled();
    return FPRounding_TIEEVEN; // keep the compiler happy
} //fp_decode_rmode

static const char * get_rmode_text( uint64_t rmode )
{
    if ( 0 == rmode )
        return "FPRounding_TIEEVEN";
    if ( 1 == rmode )
        return "FPRounding_POSINF";
    if ( 2 == rmode )
        return "FPRounding_NEGINF";
    if ( 3 == rmode )
        return "FPRounding_ZERO";
    return "unknown";
} //get_rmode_text

double Arm64::round_double( double d, FPRounding rounding )
{
    if ( FPRounding_NEGINF == rounding )
        return floor( d );
    if ( FPRounding_POSINF == rounding )
        return ceil( d );
    if ( FPRounding_TIEEVEN == rounding || FPRounding_TIEAWAY == rounding )
        return round( d );
    if ( FPRounding_ODD == rounding )
    {
        double rounded = d;
        if ( ( 0 == fmod( rounded, 2 ) ) && ( 0.5 == fabs( d - rounded ) ) )
            rounded += ( d > 0.0 ) ? 1.0 : -1.0;

        return rounded;
    }
    if ( FPRounding_ZERO == rounding )
        return (double) (int64_t) d;

    unhandled();
    return d; // keep the compiler happy
} //round_double

int32_t Arm64::double_to_fixed_int32( double d, uint64_t fracbits, FPRounding rounding )
{
    if ( d >= INT32_MAX )
        return INT32_MAX;

    if ( d <= INT32_MIN )
        return INT32_MIN;

    return (int32_t) round_double( d * ( 1ull << fracbits ), rounding );
} //double_to_fixed_int32

uint32_t Arm64::double_to_fixed_uint32( double d, uint64_t fracbits, FPRounding rounding )
{
    if ( d >= UINT32_MAX )
        return UINT32_MAX;

    double val = round_double( d * ( 1ull << fracbits ), rounding );
    if ( val < 0.0 )
        val = 0.0;
    return (uint32_t) val;
} //double_to_fixed_uint32

int64_t Arm64::double_to_fixed_int64( double d, uint64_t fracbits, FPRounding rounding )
{
    if ( d >= (double) INT64_MAX )
        return INT64_MAX;

    if ( d <= (double) INT64_MIN )
        return INT64_MIN;

    return (int64_t) round_double( d * ( 1ull << fracbits ), rounding );
} //double_to_fixed_int64

uint64_t Arm64::double_to_fixed_uint64( double d, uint64_t fracbits, FPRounding rounding )
{
    if ( d >= (double) UINT64_MAX )
        return UINT64_MAX;

    double val = round_double( d * ( 1ull << fracbits ), rounding );
    if ( val < 0.0 )
        val = 0.0;
    return (uint64_t) val;
} //double_to_fixed_uint64

static uint64_t get_bit( uint64_t x, uint64_t bit_number )
{
    assert( 64 != bit_number );
    return ( ( x >> bit_number ) & 1 );
} //get_bit

static uint64_t gen_bitmask( uint64_t n )
{
    if ( 0 == n )
        return 0;

    return ( ~0ull ) >> ( 64ull - n );
} //gen_bitmask

static uint64_t get_elem_bits( uint64_t val, uint64_t c, uint64_t container_size )
{
    uint64_t mask = gen_bitmask( container_size );
    return ( val & ( mask << ( c * container_size ) ) );
} //get_elem_bits

static uint64_t get_bits( uint64_t x, uint64_t lowbit, uint64_t len )
{
    uint64_t val = ( x >> lowbit );
    if ( 64 == len ) // this actually happens
        return val;
    return ( val & ( ( 1ull << len ) - 1 ) );
} //get_bits

static uint64_t reverse_bytes( uint64_t val, uint64_t n )
{
    uint64_t result = 0;

    if ( 8 == n )
        for ( uint64_t e = 0; e < 8; e++ )
            result |= ( get_bits( val, e * 8, 8 ) << ( ( 7 - e ) * 8 ) );
    else if ( 16 == n )
        for ( uint64_t e = 0; e < 4; e++ )
            result |= ( get_bits( val, e * 16, 16 ) << ( ( 3 - e ) * 16 ) );
    else if ( 32 == n )
        for ( uint64_t e = 0; e < 2; e++ )
            result |= ( get_bits( val, e * 32, 32 ) << ( ( 1 - e ) * 32 ) );
    else
        assert( false );

    return result;
} //reverse_bytes

static uint64_t one_bits( uint64_t bits )
{
    if ( 64 == bits )
        return ~0ull;

    return ( ( 1ull << bits ) - 1 );
} //one_bits

static uint64_t replicate_bit( uint64_t val, uint64_t len )
{
    if ( 0 != val )
        return one_bits( len );
    return 0;
} //replicate_bit

static int64_t sign_extend( uint64_t x, uint64_t high_bit )
{
    assert( high_bit < 63 );
    x &= ( 1ull << ( high_bit + 1 ) ) - 1; // clear bits above the high bit since they may contain noise
    const int64_t m = 1ull << high_bit;
    return ( x ^ m ) - m;
} //sign_extend

static uint16_t sign_extend16( uint16_t x, uint16_t high_bit )
{
    assert( high_bit < 15 );
    x &= ( 1u << ( high_bit + 1 ) ) - 1; // clear bits above the high bit since they may contain noise
    const int16_t m = ( (uint16_t) 1 ) << high_bit;
    return ( x ^ m ) - m;
} //sign_extend16

static uint32_t sign_extend32( uint32_t x, uint32_t high_bit )
{
    assert( high_bit < 31 );
    x &= ( 1u << ( high_bit + 1 ) ) - 1; // clear bits above the high bit since they may contain noise
    const int32_t m = ( (uint32_t) 1 ) << high_bit;
    return ( x ^ m ) - m;
} //sign_extend32

static uint64_t plaster_bits( uint64_t val, uint64_t bits, uint64_t low_position, uint64_t len )
{
    return ( ( val & ( ~( one_bits( len ) << low_position ) ) ) | ( bits << low_position ) );
} //plaster_bits

static uint64_t lowest_set_bit_nz( uint64_t x )
{
    uint64_t mask = 1;
    for ( uint64_t i = 0; i < 64; i++ )
    {
        if ( x & mask )
            return i;
        mask <<= 1;
    }

    assert( false ); // _nz means this shoudn't hit
    return 64;
} //lowest_set_bit_nz

static uint64_t highest_set_bit_nz( uint64_t x )
{
    uint64_t mask = 1ull << 63;
    for ( uint64_t i = 64; i > 0; i-- )
    {
        if ( x & mask )
            return i - 1;
        mask >>= 1;
    }

    assert( false ); // _nz means this shoudn't hit
    return 0;
} //highest_set_bit_nz

const char * Arm64::render_flags() const
{
    static char ac[ 5 ] = {0};

    ac[ 0 ] = fN ? 'N' : 'n';
    ac[ 1 ] = fZ ? 'Z' : 'z';
    ac[ 2 ] = fC ? 'C' : 'c';
    ac[ 3 ] = fV ? 'V' : 'v';

    return ac;
} //render_flags

static const char * xreg_names[32] =
{
    "x0",  "x1",   "x2", "x3",  "x4",  "x5",  "x6",  "x7",
    "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24", "x25", "x26", "x27", "x28", "x29", "x30", "xzr"
};

static const char * wreg_names[32] =
{
    "w0",  "w1",   "w2", "w3",  "w4",  "w5",  "w6",  "w7",
    "w8",  "w9",  "w10", "w11", "w12", "w13", "w14", "w15",
    "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23",
    "w24", "w25", "w26", "w27", "w28", "w29", "w30", "wzr"
};

static const char * reg_or_zr( uint64_t x, bool xregs )
{
    assert( x <= 31 );
    if ( xregs )
        return xreg_names[ x ];

    return wreg_names[ x ];
} //reg_or_zr

static const char * reg_or_sp( uint64_t x, bool xregs )
{
    assert( x <= 31 );
    if ( 31 == x )
        return "sp";

    return reg_or_zr( x, xregs );
} //reg_or_sp

__inline_perf uint64_t Arm64::val_reg_or_zr( uint64_t r ) const
{
    if ( 31 == r )
        return 0;
    return regs[ r ];
} //val_reg_or_zr

template < typename T > Arm64::ElementComparisonResult Arm64::compare( T l, T r )
{
    return ( l < r ) ? ecr_lt : ( l == r ) ? ecr_eq : ecr_gt;
} //compare

Arm64::ElementComparisonResult Arm64::compare_vector_elements( uint8_t * pl, uint8_t * pr, uint64_t width, bool unsigned_compare )
{
    assert( 1 == width || 2 == width || 4 == width || 8 == width );

    if ( unsigned_compare )
    {
        if ( 1 == width )
            return compare( *pl, *pr );
        if ( 2 == width )
            return compare( consider_endian16( * (uint16_t *) pl ), consider_endian16( * (uint16_t *) pr ) );
        if ( 4 == width )
            return compare( consider_endian32( * (uint32_t *) pl ), consider_endian32( * (uint32_t *) pr ) );

        return compare( consider_endian64( * (uint64_t *) pl ), consider_endian64( * (uint64_t *) pr ) );
    }

    if ( 1 == width )
        return compare( * (int8_t *) pl, * (int8_t *) pr );
    if ( 2 == width )
        return compare( (int16_t) consider_endian16( * (uint16_t *) pl ), (int16_t) consider_endian16( * (uint16_t *) pr ) );
    if ( 4 == width )
        return compare( (int32_t) consider_endian32( * (uint32_t *) pl ), (int32_t) consider_endian32( * (uint32_t *) pr ) );

    return compare( (int64_t) consider_endian64( * (uint64_t *) pl ), (int64_t) consider_endian64( * (uint64_t *) pr ) );
} //compare_vector_elements

static const char * get_ld1_vector_T( uint64_t size, uint64_t Q )
{
    if ( 0 == size )
        return Q ? "16b" : "8b";

    if ( 1 == size )
        return Q ? "8h" : "4h";

    if ( 2 == size )
        return Q ? "4s" : "2s";

    if ( 3 == size )
        return Q ? "2d" : "1d";

    return "UNKNOWN";
} //get_ld1_vector_T

static const char * get_saddlp_vector_T( uint64_t size, uint64_t Q )
{
    if ( 0 == size )
        return Q ? "8h" : "4h";

    if ( 1 == size )
        return Q ? "4s" : "2s";

    if ( 2 == size )
        return Q ? "2d" : "1d";

    return "UNKNOWN";
} //get_saddlp_vector_T

static const char * get_vector_T( uint64_t imm5, uint64_t Q )
{
    if ( 1 == ( 1 & imm5 ) )
        return Q ? "16b" : "8b";

    if ( 2 == ( 3 & imm5 ) )
        return Q ? "8h" : "4h";

    if ( 4 == ( 7 & imm5 ) )
        return Q ? "4s" : "2s";

    if ( 8 == ( 0xf & imm5 ) )
        return Q ? "2d" : "RESERVED";

    return "RESERVED";
} //get_vector_T

static const char * get_sshr_vector_T( uint64_t immh, uint64_t Q )
{
    if ( 1 == immh )
        return Q ? "16b" : "8b";

    if ( 2 == ( 0xe & immh ) )
        return Q ? "8h" : "4h";

    if ( 4 == ( 0xc & immh ) )
        return Q ? "4s" : "2s";

    if ( 8 == ( 8 & immh ) )
        return Q ? "2d" : "RESERVED";

    return "RESERVED";
} //get_sshr_vector_T

uint64_t Arm64::extend_reg( uint64_t m, uint64_t extend_type, uint64_t shift, bool fullm )
{
    uint64_t x = ( 31 == m ) ? 0 : regs[ m ];
    if ( !fullm )
        x = (uint32_t) x;

    switch( extend_type )
    {
        case 0: { x &= 0xff; break; }                 // UXTB
        case 1: { x &= 0xffff; break; }               // UXTH
        case 2: { x &= 0xffffffff; break; }           // LSL/UXTW
        case 3: { break; }                            // UXTX
        case 4: { x = sign_extend( x, 7 ); break; }   // SXTB
        case 5: { x = sign_extend( x, 15 ); break; }  // SXTH
        case 6: { x = sign_extend( x, 31 ); break; }  // SXTW
        case 7: { break; }                            // SXTX
        default: unhandled();
    }

    x <<= shift;
    return x;
} //extend_reg

static const char * shift_type( uint64_t x )
{
    if ( 0 == x )
        return "lsl";
    if ( 1 == x )
        return "lsr";
    if ( 2 == x )
        return "asr";
    if ( 3 == x )
        return "ror";

    return "UNKNOWN_SHIFT";
} //shift_type

static const char * extend_type( uint64_t x )
{
    if ( 0 == x )
        return "UXTB";
    if ( 1 == x )
        return "UXTH";
    if ( 2 == x )
        return "UXTW";
    if ( 3 == x )
        return "LSL | UXTW";
    if ( 4 == x )
        return "SXTB";
    if ( 5 == x )
        return "SXTH";
    if ( 6 == x )
        return "SXTW";
    if ( 7 == x )
        return "SXTX";
    return "UNKNOWN_EXTEND";
} //extend_type

uint64_t Arm64::replicate_bytes( uint64_t val, uint64_t byte_len )
{
    uint64_t mask = one_bits( byte_len * 8 );
    uint64_t pattern = ( val & mask );
    uint64_t repeat = 8 / byte_len;
    uint64_t result = 0;
    for ( uint64_t x = 0; x < repeat; x++ )
        result |= ( pattern << ( x * byte_len * 8 ) );

    //tracer.Trace( "replicate bytes val %#llx byte_len %lld. mask %#llx, pattern %#llx, repeat %lld, result %#llx\n", val, byte_len, mask, pattern, repeat, result );
    return result;
} //replicate_bytes

uint64_t Arm64::adv_simd_expand_imm( uint64_t operand, uint64_t cmode, uint64_t imm8 )
{
    //tracer.Trace( "operand %#llx, cmode %#llx, imm8 %#llx\n", operand, cmode, imm8 );
    uint64_t imm64 = 0;
    uint64_t cm = ( cmode >> 1 );
    switch ( cm )
    {
        case 0: { imm64 = replicate_bytes( imm8, 4 ); break; }
        case 1: { imm64 = replicate_bytes( imm8 << 8, 4 ); break; }
        case 2: { imm64 = replicate_bytes( imm8 << 16, 4 ); break; }
        case 3: { imm64 = replicate_bytes( imm8 << 24, 4 ); break; }
        case 4: { imm64 = replicate_bytes( imm8, 2 ); break; }
        case 5: { imm64 = replicate_bytes( imm8 << 8, 2 ); break; }
        case 6:
        {
            if ( 1 == ( cmode & 1 ) )
                imm64 = replicate_bytes( ( imm8 << 16 ) | 0xffff, 4 );
            else
                imm64 = replicate_bytes( ( imm8 << 8 ) | 0xff, 4 );
            break;
        }
        case 7:
        {
            if ( 0 == ( cmode & 1 ) )
            {
                if  ( 0 == operand )
                    imm64 = replicate_bytes( imm8, 1 );
                else if ( 1 == operand )
                {
                    uint64_t imm8a = ( imm8 & 0x80 ) ? 0xff : 0;
                    uint64_t imm8b = ( imm8 & 0x40 ) ? 0xff : 0;
                    uint64_t imm8c = ( imm8 & 0x20 ) ? 0xff : 0;
                    uint64_t imm8d = ( imm8 & 0x10 ) ? 0xff : 0;
                    uint64_t imm8e = ( imm8 & 0x08 ) ? 0xff : 0;
                    uint64_t imm8f = ( imm8 & 0x04 ) ? 0xff : 0;
                    uint64_t imm8g = ( imm8 & 0x02 ) ? 0xff : 0;
                    uint64_t imm8h = ( imm8 & 0x01 ) ? 0xff : 0;
                    imm64 = ( imm8a << 56 ) | ( imm8b << 48 ) | ( imm8c << 40 ) | ( imm8d << 32 ) | ( imm8e << 24 ) | ( imm8f << 16 ) | ( imm8g << 8 ) | imm8h;
                }
                else
                    unhandled();
            }
            else
            {
                if ( 0 == operand )
                {
                    // imm32 = imm8<7>:NOT(imm8<6>):Replicate(imm8<6>,5):imm8<5:0>:Zeros(19);
                    // imm64 = Replicate(imm32, 2);

                    uint64_t a = get_bit( imm8, 7 );
                    uint64_t b = !get_bit( imm8, 6 );
                    uint64_t c = replicate_bit( get_bit( imm8, 6 ), 5 );
                    uint64_t d = get_bits( imm8, 0, 6 );
                    uint32_t imm32 = (uint32_t) ( ( ( a << 12 ) | ( b << 11 ) | ( c << 6 ) | d ) << 19 );
                    imm64 = replicate_bytes( imm32, 4 );
                }
                else
                {
                    // imm64 = imm8<7>:NOT(imm8<6>):Replicate(imm8<6>,8):imm8<5:0>:Zeros(48);
                    imm64 = ( get_bit( imm8, 7 ) << 63 ) |
                            ( ( get_bit( imm8, 6 ) ? 0ull : 1ull ) << 62 ) |
                            ( replicate_bit( get_bit( imm8, 6 ), 8 ) << ( 62 - 8 ) ) |
                            ( get_bits( imm8, 0, 6 ) << 48 );
                }
            }
            break;
        }
        default: { unhandled(); }
    }

    //tracer.Trace( "expand imm cmode %#llx, from %llx to %llx\n", cmode, imm8, imm64 );
    return imm64;
} //adv_simd_expand_imm

static inline uint64_t ror( uint64_t elt, uint64_t size )
{
    return ( ( elt & 1 ) << ( size - 1 ) ) | ( elt >> 1 );
} //ror

static inline uint64_t ror_n( uint64_t elt, uint64_t size, uint64_t amount )
{
    return ( ( elt >> amount ) | ( elt << ( size - amount ) ) );
} //ror_n

static uint64_t decode_logical_immediate( uint64_t val, uint64_t bit_width )
{
    uint64_t N = get_bit( val, 12 );
    uint64_t immr = get_bits( val, 6, 6 );
    uint64_t imms = get_bits( val, 0, 6 );

    uint64_t lzero_count = count_leading_zeroes32( (uint32_t) ( ( N << 6 ) | ( ( ~imms ) & 0x3f ) ) );
    uint64_t len = 31 - lzero_count;
    uint64_t size = ( 1ull << len );
    uint64_t R = ( immr & ( size - 1 ) );
    uint64_t S = ( imms & ( size - 1 ) );
    uint64_t pattern = ( 1ull << ( S + 1 ) ) - 1;
    pattern = ror_n( pattern, size, R );

    assert( 32 == bit_width || 64 == bit_width );
    while ( size != bit_width )
    {
        pattern |= ( pattern << size );
        size *= 2;
    }

    if ( 32 == bit_width )
        pattern = (uint32_t) pattern;
    return pattern;
} //decode_logical_immediate

static const char * conditions[16] = { "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv" };

static const char * get_cond( uint64_t x )
{
    if ( x <= 15 )
        return conditions[ x ];

    return "UNKNOWN_CONDITION";
} //get_cond

static char get_byte_len( uint64_t l )
{
    if ( 1 == l )
        return 'b';
    if ( 2 == l )
        return 'h';
    if ( 4 == l )
        return 's';
    if ( 8 == l )
        return 'd';
    if ( 16 == l )
        return 'q';

    return '?';
} //get_byte_len

static uint64_t vfp_expand_imm( uint64_t imm8, uint64_t N )
{
    assert( 16 == N || 32 == N || 64 == N );
    uint64_t E = ( 16 == N ) ? 5 : ( 32 == N ) ? 8 : 11;
    uint64_t F = N - E - 1;
    uint64_t sign = ( 0 != ( imm8 & 0x80 ) );
    uint64_t exp_part_1 = ( get_bit( imm8, 6 ) ? 0 : 1 );
    uint64_t exp_part_2 = replicate_bit( get_bit( imm8, 6 ), E - 3 );
    uint64_t exp_part_3 = get_bits( imm8, 4, 2 );
    uint64_t exp = ( exp_part_1 << ( E - 3 + 2 ) ) | ( exp_part_2 << 2 ) | exp_part_3;
    uint64_t frac_shift = F - 4;
    uint64_t frac = ( imm8 & 0xf ) << frac_shift;
    uint64_t result = ( sign << ( N - 1 ) ) | ( exp << F ) | frac;
//    tracer.Trace( "result %#llx, imm8 %llu, N %llu, E %llu, F %llu, sign %llu, exp %#llx, exp1 %#llx, exp2 %#llx, exp3 %#llx, frac_shift %llu, frac %#llx\n",
//                  result, imm8, N, E, F, sign, exp, exp_part_1, exp_part_2, exp_part_3, frac_shift, frac );
    return result;
} //vfp_expand_imm

static char get_fcvt_precision( uint64_t x )
{
    return ( 0 == x ) ? 's' : ( 1 == x ) ? 'd' : ( 3 == x ) ? 'h' : '?';
} //get_fcvt_precision

void Arm64::trace_state()
{
    if ( !tracer.IsEnabled() ) // can happen when an app enables instruction tracing via a syscall but overall tracing is turned off.
        return;

    static const char * previous_symbol = 0;
    uint64_t symbol_offset;
    const char * symbol_name = emulator_symbol_lookup( pc, symbol_offset );
    if ( symbol_name == previous_symbol )
        symbol_name = "";
    else
        previous_symbol = symbol_name;

    char symbol_offset_str[ 40 ];
    symbol_offset_str[ 0 ] = 0;

    if ( 0 != symbol_name[ 0 ] )
    {
        if ( 0 != symbol_offset )
            snprintf( symbol_offset_str, _countof( symbol_offset_str ), " + %llx", symbol_offset );
        strcat( symbol_offset_str, "\n            " );
    }

    //tracer.TraceBinaryData( getmem( 0x2cb97f0 + 16 ), 16, 4 );

    tracer.Trace( "pc %8llx %s%s op %08llx %s ==> ", pc, symbol_name, symbol_offset_str, op, render_flags() );

    uint8_t hi8 = (uint8_t) ( op >> 24 );
    switch ( hi8 )
    {
        case 0: // UDF
        {
            uint64_t bits23to16 = opbits( 16, 8 );
            uint64_t imm16 = opbits( 0, 16 );
            if ( 0 == bits23to16 )
                tracer.Trace( "udf %#llx\n", imm16 );
            else
                unhandled();
            break;
        }
        case 0x5f: // SHL D<d>, D<n>, #<shift>    ;    FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]    ;    FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                   // SSHR D<d>, D<n>, #<shift>
        {
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            uint64_t opcode = opbits( 10, 6 );
            uint64_t immh = opbits( 19, 4 );
            uint64_t immb = opbits( 16, 3 );
            uint64_t bit23 = opbit( 23 );
            uint64_t bit22 = opbit( 22 );
            uint64_t bits15_10 = opbits( 10, 6 );

            if ( !bit23 && bit22 && 0x15 == opcode ) // SHL D<d>, D<n>, #<shift>
            {
                uint64_t esize = 8ull << 3;
                uint64_t shift = ( ( immh << 3 ) | immb ) - esize;
                tracer.Trace( "shl d%llu, d%llu, #%llu\n", d, n, shift );
            }
            else if ( bit23 && ( 4 == opcode || 6 == opcode || 0x24 == opcode || 0x26 == opcode ) ) // FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]    ;    FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t sz = opbit( 22 );
                uint64_t L = opbit( 21 );
                uint64_t M = opbit( 20 );
                uint64_t H = opbit( 11 );
                char v = sz ? 'd' : 's';
                uint64_t Rmhi = M;
                if ( 0 == sz )
                    Rmhi = ( H << 1 ) | L;
                else if ( 0 == L )
                    Rmhi = H;
                else
                    unhandled();

                tracer.Trace( "%s %c%llu, %c%llu, v%llu.%c[%llu]\n", ( 4 == opcode || 6 == opcode ) ? "fmla" : "fmul", v, d, v, n, m, v, Rmhi );
            }
            else if ( !bit23 && bit22 && 1 == bits15_10 ) // SSHR D<d>, D<n>, #<shift>
            {
                uint64_t esize = 8ull << 3;
                uint64_t shift = ( esize * 2 ) - ( ( immh << 3 ) | immb );
                tracer.Trace( "sshr d%llu, d%llu, #%llu\n", d, n, shift );
            }
            else
                unhandled();
            break;
        }
        case 0x0d: case 0x4d: // LD1 { <Vt>.B }[<index>], [<Xn|SP>]    ;    LD1 { <Vt>.B }[<index>], [<Xn|SP>], #1
                              // LD1R { <Vt>.<T> }, [<Xn|SP>], <imm>   ;    LD1R { <Vt>.<T> }, [<Xn|SP>], <Xm>
                              // ST1 { <Vt>.B }[<index>], [<Xn|SP>]    ;    ST1 { <Vt>.B }[<index>], [<Xn|SP>], #1
        {
            uint64_t R = opbit( 21 );
            if ( R )
                unhandled();
            uint64_t post_index = opbit( 23 );
            uint64_t opcode = opbits( 13, 3 );
            uint64_t bit13 = opbit( 13 );
            if ( bit13 )
                unhandled();
            uint64_t size = opbits( 10, 2 );
            uint64_t n = opbits( 5, 5 );
            uint64_t m = opbits( 16, 5 );
            uint64_t t = opbits( 0, 5 );
            uint64_t replicate = ( 6 == opcode );
            uint64_t S = opbit( 12 );
            uint64_t Q = opbit( 30 );
            uint64_t L = opbit( 22 );
            uint64_t index = 0;
            uint64_t scale = get_bits( opcode, 1, 2 );
            if ( 3 == scale )
                scale = size;
            else if ( 0 == scale )
                index = ( Q << 3 ) | ( S << 2 ) | size;
            else if ( 1 == scale )
                index = ( Q << 2 ) | ( S << 1 ) | get_bit( size, 1 );
            else if ( 2 == scale )
            {
                if ( 0 == ( size & 1 ) )
                    index = ( Q << 1 ) | S;
                else
                {
                    index = Q;
                    scale = 3;
                }
            }

            const char * pOP = L ? "ld" : "st";
            char type = ( 0 == opcode ) ? 'b' : ( 2 == opcode ) ? 'h' : ( 4 == opcode && 0 == size ) ? 's' : 'd';
            if ( post_index )
            {
                if ( 31 == m ) // immediate offset
                {
                    uint64_t imm = 1ull << size;
                    tracer.Trace( "%s1%s {v%llu.%c}[%llu], [%s], #%llu\n", pOP, replicate ? "r" : "", t, type, index, reg_or_sp( n, true ), imm );
                }
                else // register offset
                    tracer.Trace( "%s1%s {v%llu.%c}[%llu], [%s], %s\n", pOP, replicate ? "r" : "", t, type, index, reg_or_sp( n, true ), reg_or_zr( m, true ) );
            }
            else // no offset
                tracer.Trace( "%s1%s {v%llu.%c}[%llu], [%s]\n", pOP, replicate ? "r" : "", t, type, index, reg_or_sp( n, true ) );
            break;
        }
        case 0x08: // LDAXRB <Wt>, [<Xn|SP>{, #0}]    ;    LDARB <Wt>, [<Xn|SP>{, #0}]    ;    STLXRB <Ws>, <Wt>, [<Xn|SP>{, #0}]    ;    STLRB <Wt>, [<Xn|SP>{, #0}]
                   // STXRB <Ws>, <Wt>, [<Xn|SP>{, #0}] ;  LDXRB <Wt>, [<Xn|SP>{, #0}]
        case 0x48: // LDAXRH <Wt>, [<Xn|SP>{, #0}]    ;    LDARH <Wt>, [<Xn|SP>{, #0}]    ;    STLXRH <Ws>, <Wt>, [<Xn|SP>{, #0}]    ;    STLRH <Wt>, [<Xn|SP>{, #0}]
                   // STXRH <Ws>, <Wt>, [<Xn|SP>{, #0}] ;  LDXRH <Wt>, [<Xn|SP>{, #0}]
        {
            uint64_t bit23 = opbit( 23 );
            uint64_t L = opbit( 21 );
            uint64_t bit21 = opbit( 21 );
            uint64_t s = opbits( 16, 5 );
            uint64_t oO = opbit( 15 );
            uint64_t t2 = opbits( 10, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t t = opbits( 0, 5 );

            if ( 0 != bit21 || 0x1f != t2 )
                unhandled();

            char suffix = ( hi8 & 0x40 ) ? 'h' : 'b';

            if ( L )
            {
                if ( 0x1f != s )
                    unhandled();
                tracer.Trace( "%s%c, w%llu, [%s, #0]\n", bit23 ? "ldar" : oO ? "ldaxr" : "ldxr", suffix, t, reg_or_sp( n, true ) );
            }
            else
            {
                if ( bit23 )
                    tracer.Trace( "stlr%c w%llu, [%s, #0]\n", suffix, t, reg_or_sp( n, true ) );
                else
                    tracer.Trace( "%s%c w%llu, w%llu, [%s, #0]\n", oO ? "stlxr" : "stxr", suffix, s, t, reg_or_sp( n, true ) );
            }
            break;
        }
        case 0x1f: // fmadd, vnmadd, fmsub, fnmsub
        {
            uint64_t ftype = opbits( 22, 2 );
            uint64_t bit21 = opbit( 21 );
            uint64_t bit15 = opbit( 15 );
            uint64_t m = opbits( 16, 5 );
            uint64_t a = opbits( 10, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );

            bool isn = ( 0 != bit21 );
            char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
            if ( !bit15 )
                tracer.Trace( "%s %c%llu, %c%llu, %c%llu, %c%llu\n", isn ? "fnmadd" : "fmadd", t, d, t, n, t, m, t, a );
            else
                tracer.Trace( "%s %c%llu, %c%llu, %c%llu, %c%llu\n", isn ? "fnmsub" : "fmsub", t, d, t, n, t, m, t, a );
            break;
        }
        case 0x3c: // LDR <Bt>, [<Xn|SP>], #<simm>    ;    LDR <Bt>, [<Xn|SP>, #<simm>]!    ;    LDR <Qt>, [<Xn|SP>], #<simm>    ;     LDR <Qt>, [<Xn|SP>, #<simm>]!
        case 0x3d: // LDR <Bt>, [<Xn|SP>{, #<pimm>}]  ;    LDR <Qt>, [<Xn|SP>{, #<pimm>}]
        case 0x7c: // LDR <Ht>, [<Xn|SP>], #<simm>    ;    LDR <Ht>, [<Xn|SP>, #<simm>]!
        case 0x7d: // LDR <Ht>, [<Xn|SP>{, #<pimm>}]
        case 0xbc:
        case 0xbd: // LDR <Dt>, [<Xn|SP>{, #<pimm>}]
        case 0xfc: // LDR <Dt>, [<Xn|SP>], #<simm>    ;    LDR <Dt>, [<Xn|SP>, #<simm>]!    ;    STR <Dt>, [<Xn|SP>], #<simm>    ;    STR <Dt>, [<Xn|SP>, #<simm>]!
        case 0xfd: // LDR <Dt>, [<Xn|SP>{, #<pimm>}]  ;    STR <Dt>, [<Xn|SP>{, #<pimm>}]   ;    STR <Dt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
        {
            uint64_t bits11_10 = opbits( 10, 2 );
            uint64_t bit21 = opbit( 21 );
            bool unsignedOffset = ( 0xd == ( hi8 & 0xf ) );
            bool preIndex = ( ( 0xc == ( hi8 & 0xf ) ) && ( 3 == bits11_10 ) );
            bool postIndex = ( ( 0xc == ( hi8 & 0xf ) ) && ( 1 == bits11_10 ) );
            bool signedUnscaledOffset = ( ( 0xc == ( hi8 & 0xf ) ) && ( 0 == bits11_10 ) );
            bool shiftExtend = ( ( 0xc == ( hi8 & 0xf ) ) && ( bit21 ) && ( 2 == bits11_10 ) );
            uint64_t imm12 = opbits( 10, 12 );
            int64_t imm9 = sign_extend( opbits( 12, 9 ), 8 );
            uint64_t size = opbits( 30, 2 );
            uint64_t opc = opbits( 22, 2 );
            bool is_ldr = opbit( 22 );
            uint64_t t = opbits( 0, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t byte_len = 1ull << size;

            if ( is_ldr )
            {
                if ( 3 == opc )
                    byte_len = 16;

                if ( preIndex )
                    tracer.Trace( "ldr %c%llu, [%s, #%lld]! //pr\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), imm9 );
                else if ( postIndex )
                    tracer.Trace( "ldr %c%llu, [%s] #%lld //po\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), imm9 );
                else if ( unsignedOffset )
                    tracer.Trace( "ldr %c%llu, [%s, #%llu] //uo\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), imm12 * byte_len );
                else if ( signedUnscaledOffset )
                    tracer.Trace( "ldur %c%llu, [%s, #%lld] //so\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), imm9 );
                else if ( shiftExtend )
                {
                    uint64_t option = opbits( 13, 3 );
                    uint64_t m = opbits( 16, 5 );
                    uint64_t shift = 0;
                    uint64_t S = opbit( 12 );
                    if ( 0 != S )
                    {
                        if ( 0 == size )
                        {
                           if ( 3 == opc )
                               shift = 4;
                           else if ( 1 != opc )
                               unhandled();
                        }
                        else if ( 1 == size && 1 == opc )
                            shift = 1;
                        else if ( 2 == size && 1 == opc )
                            shift = 2;
                        else if ( 3 == size && 1 == opc )
                            shift = 3;
                        else
                            unhandled();
                    }

                    tracer.Trace( "ldr %c%llu, [%s, %s, %s, #%lld] //se\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), reg_or_zr( m, true ),
                                   extend_type( option ), shift );
                }
                else
                    unhandled();
            }
            else // str
            {
                if ( 2 == opc )
                    byte_len = 16;

                if ( preIndex )
                    tracer.Trace( "str %c%llu, [%s, #%lld]! //pr\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), imm9 );
                else if ( postIndex )
                    tracer.Trace( "str %c%llu, [%s] #%lld //po\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), imm9 );
                else if ( unsignedOffset )
                    tracer.Trace( "str %c%llu, [%s, #%llu] //uo\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), imm12 * byte_len );
                else if ( signedUnscaledOffset )
                    tracer.Trace( "stur %c%llu, [%s, #%lld] //so\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), imm9 );
                else if ( shiftExtend )
                {
                    uint64_t option = opbits( 13, 3 );
                    uint64_t m = opbits( 16, 5 );
                    uint64_t shift = 0;
                    uint64_t S = opbit( 12 );
                    if ( 0 != S )
                    {
                        if ( 0 == size )
                        {
                           if ( 2 == opc )
                               shift = 4;
                           else if ( 0 != opc )
                               unhandled();
                        }
                        else if ( 1 == size && 0 == opc )
                            shift = 1;
                        else if ( 2 == size && 0 == opc )
                            shift = 2;
                        else if ( 3 == size && 0 == opc )
                            shift = 3;
                        else
                            unhandled();
                    }

                    tracer.Trace( "str %c%llu, [%s, %s, %s, #%lld] //se\n", get_byte_len( byte_len ), t, reg_or_sp( n, true ), reg_or_zr( m, true ),
                                   extend_type( option ), shift );
                }
                else
                    unhandled();
            }
            break;
        }
        case 0x2c: // STP <St1>, <St2>, [<Xn|SP>], #<imm>     ;    LDP <St1>, <St2>, [<Xn|SP>], #<imm>
        case 0x6c: // STP <Dt1>, <Dt2>, [<Xn|SP>], #<imm>     ;    LDP <Dt1>, <Dt2>, [<Xn|SP>], #<imm>
        case 0xac: // STP <Qt1>, <Qt2>, [<Xn|SP>], #<imm>          LDP <Qt1>, <Qt2>, [<Xn|SP>], #<imm>
        case 0x2d: // STP <St1>, <St2>, [<Xn|SP>, #<imm>]!    ;    STP <St1>, <St2>, [<Xn|SP>{, #<imm>}]    ;    LDP <St1>, <St2>, [<Xn|SP>, #<imm>]!    ;    LDP <St1>, <St2>, [<Xn|SP>{, #<imm>}]
        case 0x6d: // STP <Dt1>, <Dt2>, [<Xn|SP>, #<imm>]!    ;    STP <Dt1>, <Dt2>, [<Xn|SP>{, #<imm>}]    ;    LDP <Dt1>, <Dt2>, [<Xn|SP>, #<imm>]!    ;    LDP <Dt1>, <Dt2>, [<Xn|SP>{, #<imm>}]
        case 0xad: // STP <Qt1>, <Qt2>, [<Xn|SP>, #<imm>]!    ;    STP <Qt1>, <Qt2>, [<Xn|SP>{, #<imm>}]    ;    LDP <Qt1>, <Qt2>, [<Xn|SP>, #<imm>]!    ;    LDP <Qt1>, <Qt2>, [<Xn|SP>{, #<imm>}]
        {
            uint64_t opc = opbits( 30, 2 );
            char vector_width = ( 0 == opc ) ? 's' : ( 1 == opc ) ? 'd' : 'q';
            uint64_t imm7 = opbits( 15, 7 );
            uint64_t t2 = opbits( 10, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t t1 = opbits( 0, 5 );
            uint64_t L = opbit( 22 );
            uint64_t bit23 = opbit( 23 );

            bool preIndex = ( ( 0xd == ( hi8 & 0xf ) ) && bit23 );
            bool postIndex = ( ( 0xc == ( hi8 & 0xf ) ) && bit23 );
            bool signedOffset = ( ( 0xd == ( hi8 & 0xf ) ) && !bit23 );

            uint64_t scale = 2 + opc;
            int64_t offset = sign_extend( imm7, 6 ) << scale;

            if ( L )
            {
                if ( postIndex )
                   tracer.Trace( "ldp %c%llu, %c%llu, [%s], #%lld //po\n", vector_width, t1, vector_width, t2, reg_or_sp( n, true ), offset );
                else if ( preIndex )
                   tracer.Trace( "ldp %c%llu, %c%llu, [%s, #%lld]! //pr\n", vector_width, t1, vector_width, t2, reg_or_sp( n, true ), offset );
                else if ( signedOffset )
                   tracer.Trace( "ldp %c%llu, %c%llu, [%s, #%lld] //so\n", vector_width, t1, vector_width, t2, reg_or_sp( n, true ), offset );
                else
                    unhandled();
            }
            else
            {
                if ( postIndex )
                   tracer.Trace( "stp %c%llu, %c%llu, [%s], #%lld //po\n", vector_width, t1, vector_width, t2, reg_or_sp( n, true ), offset );
                else if ( preIndex )
                   tracer.Trace( "stp %c%llu, %c%llu, [%s, #%lld]! //pr\n", vector_width, t1, vector_width, t2, reg_or_sp( n, true ), offset );
                else if ( signedOffset )
                   tracer.Trace( "stp %c%llu, %c%llu, [%s, #%lld] //so\n", vector_width, t1, vector_width, t2, reg_or_sp( n, true ), offset );
                else
                    unhandled();
            }
            break;
        }
        case 0x0f: case 0x2f: case 0x4f: case 0x6f: case 0x7f:
            // BIC <Vd>.<T>, #<imm8>{, LSL #<amount>}    ;    MOVI <Vd>.<T>, #<imm8>{, LSL #0}    ;          MVNI <Vd>.<T>, #<imm8>, MSL #<amount>
            // USHR <Vd>.<T>, <Vn>.<T>, #<shift>         ;    FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]  ; MVNI <Vd>.<T>, #<imm8>{, LSL #<amount>}
            // FMOV <Vd>.<T>, #<imm>                     ;    FMOV <Vd>.<T>, #<imm>               ;    FMOV <Vd>.2D, #<imm>
            // USHLL{2} <Vd>.<Ta>, <Vn>.<Tb>, #<shift>   ;    SHRN{2} <Vd>.<Tb>, <Vn>.<Ta>, #<shift>  ;   SSHLL{2} <Vd>.<Ta>, <Vn>.<Tb>, #<shift>
            // FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>] ;  SSHR <Vd>.<T>, <Vn>.<T>, #<shift>   ;    SHL <Vd>.<T>, <Vn>.<T>, #<shift>
            // MUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>] ;   URSRA <Vd>.<T>, <Vn>.<T>, #<shift>  ;    USRA <Vd>.<T>, <Vn>.<T>, #<shift>
            // ORR <Vd>.<T>, #<imm8>{, LSL #<amount>}    ;    MOVI <Vd>.<T>, #<imm8>, MSL #<amount>  ; 
        {
            uint64_t cmode = opbits( 12, 4 );
            uint64_t abc = opbits( 16, 3 );
            uint64_t defgh = opbits( 5, 5 );
            uint64_t val = ( abc << 5 ) | defgh;
            uint64_t Q = opbit( 30 );
            uint64_t bit29 = opbit( 29 );
            uint64_t o2 = opbits( 11,1 );
            uint64_t bit10 = opbit( 10 );
            uint64_t bit11 = opbit( 11 );
            uint64_t bit12 = opbit( 12 );
            uint64_t bit23 = opbit( 23 );
            uint64_t d = opbits( 0, 5 );
            uint64_t bits23_19 = opbits( 19, 5 );
            uint64_t imm = adv_simd_expand_imm( bit29, cmode, val );
            //tracer.Trace( "bit12: %llu, cmode %llx, bit29 %llu, bit11 %llu, bit10 %llu\n", bit12, cmode, bit29, bit11, bit10 );

            if ( 0 == bits23_19 )
            {
                if ( ( 0x4f == hi8 || 0x0f == hi8 ) && ( 0xc == ( cmode & 0xe ) ) && !bit11 && bit10 ) // MOVI <Vd>.<T>, #<imm8>, MSL #<amount>
                {
                    const char * pT = Q ? "4s" : "2s";
                    uint64_t amount = ( cmode & 1 ) ? 16 : 8;
                    tracer.Trace( "movi v%llu.%s, #%llu, msl #%llu\n", d, pT, imm, amount );
                }
                else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && ( 9 == ( 0xd & cmode ) || 1 == ( 9 & cmode ) ) && bit12 && !bit11 && bit10 ) // ORR <Vd>.<T>, #<imm8>{, LSL #<amount>}
                {
                    if ( 9 == ( 0xd & cmode ) ) // 16-bit variant
                    {
                        const char * pT = Q ? "8h" : "4h";
                        uint64_t amount = ( cmode & 2 ) ? 8 : 0;
                        tracer.Trace( "orr v%llu.%s, #%#llx, LSL #%llu\n", d, pT, val, amount );
                    }
                    else if ( 1 == ( 9 & cmode ) ) // 32-bit variant
                    {
                        const char * pT = Q ? "4s" : "2s";
                        uint64_t amount = 8 * get_bits( cmode, 1, 2 );
                        tracer.Trace( "orr v%llu.%s, #%#llx, LSL #%llu\n", d, pT, val, amount );
                    }
                    else
                        unhandled();
                }
                else if ( ( 0x2f == hi8 || 0x6f == hi8 ) && !bit11 && bit10 &&
                     ( ( 8 == ( cmode & 0xd ) ) || ( 0 == ( cmode & 9 ) ) || ( 0xc == ( cmode & 0xe ) ) ) ) // mvni
                {
                    if ( 8 == ( cmode & 0xd ) ) // 16-bit shifted immediate
                    {
                        uint64_t amount = ( cmode & 2 ) ? 8 : 0;
                        const char * pT = Q ? "8H" : "4H";
                        tracer.Trace( "mvni v%llu.%s, #%#llx, lsl #%llu\n", d, pT, val, amount );
                    }
                    else if ( 0 == ( cmode & 9 ) ) // 32-bit shifted immediate
                    {
                        uint64_t amount = get_bits( cmode, 1, 2 ) * 8;
                        const char * pT = Q ? "4S" : "2S";
                        tracer.Trace( "mvni v%llu.%s, #%#llx, lsl #%llu\n", d, pT, val, amount );
                    }
                    else if ( 0xc == ( cmode & 0xe ) ) // 32-bit shifting ones
                    {
                        imm = adv_simd_expand_imm( 1, cmode, val );
                        uint64_t amount = get_bit( cmode, 0 ) ? 16 : 8;
                        const char * pT = Q ? "4S" : "2S";
                        tracer.Trace( "mvni v%llu.%s, #%#llx, msl #%llu\n", d, pT, val, amount );
                    }
                    else
                        unhandled();
                }
                else if ( !bit12 || ( 0xc == ( cmode & 0xe ) ) ) // movi
                {
                    if ( !bit29 ) // movi
                    {
                        if ( 0xe == cmode )
                        {
                            const char * pT = Q ? "16B" : "8B";
                            tracer.Trace( "movi v%llu.%s, #%#llx // imm %llx\n", d, pT, val, imm );
                        }
                        else if ( 8 == ( cmode & 0xd ) )
                        {
                            const char * pT = Q ? "8H" : "4H";
                            uint64_t amount = ( cmode & 2 ) ? 8 : 0;
                            tracer.Trace( "movi v%llu.%s, #%#llx, lsl #%llu\n", d, pT, imm, amount );
                        }
                        else if ( 0 == ( cmode & 9 ) )
                        {
                            const char * pT = Q ? "4S" : "2S";
                            uint64_t amount = ( 8 * ( ( cmode >> 1 ) & 3 ) );
                            tracer.Trace( "movi v%llu.%s, #%#llx, lsl #%llu\n", d, pT, imm, amount );
                        }
                        else if ( 0xa == ( cmode & 0xe ) )
                        {
                            const char * pT = Q ? "4S" : "2S";
                            uint64_t amount = ( cmode & 1 ) ? 16 : 8;
                            tracer.Trace( "movi v%llu.%s, #%#llx, msl #%llu\n", d, pT, imm, amount );
                        }
                        else
                            unhandled();
                    }
                    else // movi
                    {
                        uint64_t a = opbit( 18 );
                        uint64_t b = opbit( 17 );
                        uint64_t c = opbit( 16 );
                        uint64_t bitd = opbit( 9 );
                        uint64_t e = opbit( 8 );
                        uint64_t f = opbit( 7 );
                        uint64_t g = opbit( 6 );
                        uint64_t h = opbit( 5 );

                        imm = a ? ( 0xffull << 56 ) : 0;
                        imm |= b ? ( 0xffull << 48 ) : 0;
                        imm |= c ? ( 0xffull << 40 ) : 0;
                        imm |= bitd ? ( 0xffull << 32 ) : 0;
                        imm |= e ? ( 0xffull << 24 ) : 0;
                        imm |= f ? ( 0xffull << 16 ) : 0;
                        imm |= g ? ( 0xffull << 8 ) : 0;
                        imm |= h ? 0xffull : 0;

                        //tracer.Trace( "movi bit29 must be 1, Q %llu, cmode %llu\n", Q, cmode );

                        if ( ( 0 == Q ) && ( cmode == 0xe ) )
                            tracer.Trace( "movi D%llu, #%#llx\n", d, imm );
                        else if ( ( 1 == Q ) && ( cmode == 0xe ) )
                            tracer.Trace( "movi V%llu.2D, #%#llx\n", d, imm );
                        else
                            unhandled();
                    }
                }
                else if ( ( 0x6f == hi8 || 0x4f == hi8 || 0x2f == hi8 || 0x0f == hi8 ) && 0xf == cmode && !bit11 && bit10 ) // fmov single and double precision immediate
                {
                    double dval = 0.0;
                    tracer.Trace( "imm6: %#llx\n", imm );
                    if ( bit29 )
                        mcpy( &dval, &imm, sizeof( dval ) );
                    else
                    {
                        float float_val;
                        mcpy( &float_val, &imm, sizeof( float_val ) );
                        dval = (double) float_val;
                    }
                    tracer.Trace( "fmov v%llu.%s, #%lf\n", d, bit29 ? "2D" : Q ? "4S" : "2S", dval );
                }
                else if ( !bit29 ) // BIC register
                {
                    unhandled();
                }
                else if ( bit29 && bit12 ) // BIC immediate
                {
                    if ( 0 != o2 || 1 != bit10 )
                        unhandled();

                    bool sixteen_bit_mode = ( cmode == 0x9 || cmode == 0xb );
                    const char * pT = "";
                    uint64_t amount = 0;
                    if ( sixteen_bit_mode )
                    {
                        pT = Q ? "8H" : "4H";
                        amount = ( cmode & 2 ) ? 8 : 0;
                    }
                    else
                    {
                        pT = Q ? "4S" : "2S";
                        amount = 8 * ( ( cmode >> 1 ) & 3 );
                    }

                    tracer.Trace( "bic v%llu.%s, #%#llx, lsl #%llu\n", d, pT, val, amount );
                    //tracer.Trace( "bic bonus: cmode %#llx, val: %#llx, abc %#llx, defgh %#llx, sixteen_bit_mode %d, imm %#llx\n", cmode, val, abc, defgh, sixteen_bit_mode, imm );
                }
            }
            else // USHR, USHLL, SHRN, SHRN2, etc
            {
                uint64_t opcode = opbits( 12, 4 );
                uint64_t bits23_22 = opbits( 22, 2 );
                uint64_t bits15_12 = opbits( 12, 4 );
                uint64_t immh = opbits( 19, 4 );

                if ( ( 0x2f == hi8 || 0x6f == hi8 ) && !bit23 && 0 != immh && 1 == bits15_12 && !bit11 && bit10 ) // USRA <Vd>.<T>, <Vn>.<T>, #<shift>
                {
                    uint64_t immb = opbits( 16, 3 );
                    uint64_t n = opbits( 5, 5 );
                    uint64_t esize = 8ull << highest_set_bit_nz( immh );
                    uint64_t shift = ( esize * 2 ) - ( ( immh << 3 ) | immb );
                    const char * pT = get_sshr_vector_T( immh, Q );
                    tracer.Trace( "usra v%llu.%s, v%llu.%s, #%llu\n", d, pT, n, pT, shift );
                }
                else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && !bit23 && 0 != immh && 1 == bits15_12 && !bit11 && bit10 ) // SSRA <Vd>.<T>, <Vn>.<T>, #<shift>
                {
                    uint64_t immb = opbits( 16, 3 );
                    uint64_t n = opbits( 5, 5 );
                    uint64_t esize = 8ull << highest_set_bit_nz( immh );
                    uint64_t shift = ( esize * 2 ) - ( ( immh << 3 ) | immb );
                    const char * pT = get_sshr_vector_T( immh, Q );
                    tracer.Trace( "ssra v%llu.%s, v%llu.%s, #%llu\n", d, pT, n, pT, shift );
                }
                else if ( ( ( 1 == bits23_22 || 2 == bits23_22 ) && !bit10 ) &&
                     ( ( ( 0x4f == hi8 || 0x0f == hi8 ) && 8 == bits15_12 ) ||    // MUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                       ( ( 0x2f == hi8 || 0x6f == hi8 ) && 0 == bits15_12 ) ) )   // MLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                {
                    uint64_t size = bits23_22;
                    uint64_t n = opbits( 5, 5 );
                    uint64_t L = opbit( 21 );
                    uint64_t M = opbit( 20 );
                    uint64_t H = opbit( 11 );
                    uint64_t index = 0;
                    uint64_t rmhi = 0;
                    char TS = '?';
                    const char * pT = "invalid";
                    if ( 1 == size )
                    {
                        index = ( H << 2 ) | ( L << 1 ) | M;
                        rmhi = 0;
                        TS = 'h';
                        pT = Q ? "8h" : "4h";
                    }
                    else if ( 2 == size )
                    {
                        index = ( H << 1 ) | L;
                        rmhi = M;
                        TS = 's';
                        pT = Q ? "4s" : "2s";
                    }
                    else
                        unhandled();

                    uint64_t m = ( rmhi << 4 ) | opbits( 16, 4 );
                    tracer.Trace( "%s v%llu.%s, v%llu.%s, v%llu.%c[%llu]\n", ( 0x2f == hi8 || 0x6f == hi8 ) ? "mla" : "mul", d, pT, n, pT, m, TS, index );
                }
                else if ( ( 0x4f == hi8 || 0x0f == hi8 ) && !bit23 && 5 == opcode && !bit11 && bit10 ) // SHL <Vd>.<T>, <Vn>.<T>, #<shift>
                {
                    uint64_t n = opbits( 5, 5 );
                    uint64_t immb = opbits( 16, 3 );
                    uint64_t esize = 8ull << highest_set_bit_nz( immh );
                    uint64_t shift = ( ( immh << 3 ) | immb ) - esize;
                    const char * pT = get_sshr_vector_T( immh, Q );
                    tracer.Trace( "shl v%llu.%s, v%llu.%s, #%llu\n", d, pT, n, pT, shift );
                }
                else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && !bit23 && 0 == opcode && !bit11 && bit10 ) // SSHR <Vd>.<T>, <Vn>.<T>, #<shift>
                {
                    uint64_t n = opbits( 5, 5 );
                    uint64_t immb = opbits( 16, 3 );
                    uint64_t esize = 8ull << highest_set_bit_nz( immh );
                    uint64_t shift = ( esize * 2 ) - ( ( immh << 3 ) | immb );
                    const char * pT = get_sshr_vector_T( immh, Q );
                    tracer.Trace( "sshr v%llu.%s, v%llu.%s, #%llu\n", d, pT, n, pT, shift );
                }
                else if ( ( 0x4f == hi8 || 0x0f == hi8 ) && bit23 && 1 == opcode && !bit10 ) // FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                {
                    uint64_t n = opbits( 5, 5 );
                    uint64_t m = opbits( 16, 5 );
                    uint64_t sz = opbit( 22 );
                    uint64_t L = opbit( 21 );
                    uint64_t H = opbit( 11 );
                    uint64_t szL = ( sz << 1 ) | L;
                    uint64_t index = ( 0 == sz ) ? ( ( H << 1 ) | L ) : ( 2 == szL ) ? H : 0;
                    uint64_t Qsz = ( Q << 1 ) | sz;
                    const char * pT = ( 0 == Qsz ) ? "2s" : ( 2 == Qsz ) ? "4s" : ( 3 == Qsz ) ? "2d" : "?";
                    tracer.Trace( "fmla v%llu.%s, v%llu.%s, v%llu.%c[%llu]\n", d, pT, n, pT, m, sz ? 'd' : 's', index );
                }
                else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && !bit23 && 0 != bits23_19 && 0xa == opcode && !bit11 && bit10 ) // SSHLL{2} <Vd>.<Ta>, <Vn>.<Tb>, #<shift>
                {
                    uint64_t n = opbits( 5, 5 );
                    uint64_t immb = opbits( 16, 3 );
                    uint64_t esize = 8ull << highest_set_bit_nz( immh & 0x7 );
                    uint64_t shift = ( ( immh << 3 ) | immb ) - esize;
                    const char * pTA = ( 1 == immh ) ? "8H" : ( 2 == ( 0xe & immh ) ) ? "4S" : "2D";
                    uint64_t sizeb = immh >> 1;
                    if ( 4 & sizeb )
                        sizeb = 4;
                    else if ( 2 & sizeb )
                        sizeb = 2;
                    const char * pTB = get_ld1_vector_T( sizeb, Q );
                    tracer.Trace( "sshll%s v%llu.%s, v%llu.%s, #%llu\n", Q ? "2" : "", d, pTA, n, pTB, shift );
                }
                else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && !bit23 && 0 != bits23_19 && 8 == opcode && !bit11 && bit10 ) // SHRN{2} <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
                {
                    uint64_t n = opbits( 5, 5 );
                    uint64_t immb = opbits( 16, 3 );
                    uint64_t esize = 8ull << highest_set_bit_nz( immh & 0x7 );
                    uint64_t shift = ( 2 * esize ) - ( ( immh << 3 ) | immb );
                    const char * pTA = ( 1 == immh ) ? "8H" : ( 2 == ( 0xe & immh ) ) ? "4S" : "2D";
                    uint64_t sizeb = immh >> 1;
                    if ( 4 & sizeb )
                        sizeb = 4;
                    else if ( 2 & sizeb )
                        sizeb = 2;
                    const char * pTB = get_ld1_vector_T( sizeb, Q );
                    tracer.Trace( "shrn%s v%llu.%s, v%llu.%s, #%llu\n", Q ? "2" : "", d, pTB, n, pTA, shift );
                }
                else if ( ( 0x2f == hi8 || 0x6f == hi8 ) && !bit23 && 0 != bits23_19 && ( 0xa == opcode ) && !bit11 && bit10 ) // USHLL{2} <Vd>.<Ta>, <Vn>.<Tb>, #<shift>
                {
                    uint64_t n = opbits( 5, 5 );
                    uint64_t immb = opbits( 16, 3 );
                    uint64_t esize = 8ull << highest_set_bit_nz( immh & 0x7 );
                    if ( 0x7f == hi8 )
                        esize = 8ull << 3;
                    uint64_t shift = ( ( immh << 3 ) | immb ) - esize;
                    const char * pTA = ( 1 == immh ) ? "8H" : ( 2 == ( 0xe & immh ) ) ? "4S" : "2D";
                    uint64_t sizeb = immh >> 1;
                    if ( 2 & sizeb )
                        sizeb = 2;
                    const char * pTB = get_ld1_vector_T( sizeb, Q );
                    tracer.Trace( "ushll%s v%llu.%s, v%llu.%s, #%llu\n", Q ? "2" : "", d, pTA, n, pTB, shift );
                }
                else if ( ( 0x2f == hi8 || 0x7f == hi8 || 0x6f == hi8 ) && !bit23 && 0 == opcode && !bit11 && bit10 ) // USHR
                {
                    uint64_t n = opbits( 5, 5 );
                    uint64_t immb = opbits( 16, 3 );
                    uint64_t esize = 8ull << highest_set_bit_nz( immh );
                    if ( 0x7f == hi8 )
                        esize = 8ull << 3;
                    uint64_t shift = ( esize * 2 ) - ( ( immh << 3 ) | immb );
                    tracer.Trace( "immh %llx, Q %llx\n", immh, Q );
                    uint64_t p_type = 0;
                    if ( 8 & immh )
                        p_type = 3;
                    else if ( 4 & immh )
                        p_type = 2;
                    else if ( 2 & immh )
                        p_type = 1;
                    else if ( 1 & immh )
                        p_type = 0;
                    const char * pT = get_ld1_vector_T( p_type, Q );
                    if ( 0x7f == hi8 ) // USHR D<d>, D<n>, #<shift>
                        tracer.Trace( "ushr, d%llu, d%llu, #%llu\n", d, n, shift );
                    else // USHR <Vd>.<T>, <Vn>.<T>, #<shift>
                        tracer.Trace( "ushr, v%llu.%s, v%llu.%s, #%llu\n", d, pT, n, pT, shift );
                }
                else if ( bit23 && !bit10 && 9 == opcode ) // FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]. Vector, single-precision and double-precision
                {
                    uint64_t n = opbits( 5, 5 );
                    uint64_t m = opbits( 16, 5 );
                    uint64_t sz = opbit( 22 );
                    uint64_t L = opbit( 21 );
                    uint64_t H = opbit( 11 );
                    uint64_t index = ( !sz ) ? ( ( H << 1 ) | L ) : H;
                    const char * pT = ( Q && sz ) ? "2D" : ( !Q && !sz ) ? "2S" : ( Q && !sz ) ? "4S" : "?";
                    tracer.Trace( "fmul v%llu.%s, v%llu.%s, v%llu.%c[%llu]\n", d, pT, n, pT, m, sz ? 'D' : 'S', index );
                }
                else
                    unhandled();
            }
            break;
        }
        case 0x5a: // REV <Wd>, <Wn>  ;  CSINV <Wd>, <Wn>, <Wm>, <cond>  ;  RBIT <Wd>, <Wn>  ;  CLZ <Wd>, <Wn>  ;  CSNEG <Wd>, <Wn>, <Wm>, <cond>  ;  SBC <Wd>, <Wn>, <Wm> ; REV16 <Wd>, <Wn>
        case 0xda: // REV <Xd>, <Xn>  ;  CSINV <Xd>, <Xn>, <Xm>, <cond>  ;  RBIT <Xd>, <Xn>  ;  CLZ <Xd>, <Xn>  ;  CSNEG <Xd>, <Xn>, <Xm>, <cond>  ;  SBC <Xd>, <Xn>, <Xm> ; REV16 <Xd>, <Xn> ; REV32 <Xd>, <Xn>
        {
            uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
            uint64_t bits23_21 = opbits( 21, 3 );
            uint64_t bits23_10 = opbits( 10, 14 );
            uint64_t bits15_10 = opbits( 10, 6 );
            uint64_t bit11 = opbit( 11 );
            uint64_t bit10 = opbit( 10 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );

            if ( 0xda == hi8 && 0x3002 == bits23_10 ) // rev32
                tracer.Trace( "rev32 %s, %s\n", reg_or_zr( d, true ), reg_or_zr( n, true ) );
            else if ( 0x3001 == bits23_10 ) // rev16
                tracer.Trace( "rev16 %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ) );
            else if ( 4 == bits23_21 ) // csinv + csneg
            {
                if ( bit11 )
                    unhandled();
                uint64_t m = opbits( 16, 5 );
                uint64_t cond = opbits( 12, 4 );
                tracer.Trace( "%s %s, %s, %s, %s\n", bit10 ? "csneg" : "csinv", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ),  reg_or_zr( m, xregs ), get_cond( cond ) );
            }
            else if ( 6 == bits23_21 )
            {
                if ( 0 == bits15_10 ) // rbit
                    tracer.Trace( "rbit %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ) );
                else if ( 2 == bits15_10 || 3 == bits15_10 ) // rev
                    tracer.Trace( "rev %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ) );
                else if ( 4 == bits15_10 ) // clz
                    tracer.Trace( "clz %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ) );
                else
                    unhandled();
            }
            else if ( 0 == bits23_21 )
            {
                if ( 0 == bits15_10 ) // sbc
                {
                    uint64_t m = opbits( 16, 5 );
                    tracer.Trace( "sbc %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
                }
                else
                    unhandled();
            }
            else
                unhandled();
            break;
        }
        case 0x14: case 0x15: case 0x16: case 0x17: // b label
        {
            int64_t imm26 = opbits( 0, 26 );
            imm26 <<= 2;
            imm26 = sign_extend( imm26, 27 );
            tracer.Trace( "b %#llx\n", pc + imm26 );
            break;
        }
        case 0x54: // b.cond
        {
            uint64_t cond = opbits( 0, 4 );
            int64_t imm19 = opbits( 5, 19 );
            imm19 <<= 2;
            imm19 = sign_extend( imm19, 20 );
            tracer.Trace( "b.%s %#llx\n", get_cond( cond ), pc + imm19 );
            break;
        }
        case 0x18: // ldr wt, (literal)
        case 0x58: // ldr xt, (literal)
        {
            uint64_t imm19 = opbits( 5, 19 );
            uint64_t t = opbits( 0, 5 );
            bool xregs = ( 0 != opbit( 30 ) );
            tracer.Trace( "ldr %s, =%#llx\n", reg_or_zr( t, xregs ), pc + ( imm19 << 2 ) );
            break;
        }
        case 0x3a: // CCMN <Wn>, #<imm>, #<nzcv>, <cond>  ;    CCMN <Wn>, <Wm>, #<nzcv>, <cond>       ;    ADCS <Wd>, <Wn>, <Wm>
        case 0xba: // CCMN <Wn>, <Wm>, #<nzcv>, <cond>    ;    CCMN <Xn>, <Xm>, #<nzcv>, <cond>       ;    ADCS <Xd>, <Xn>, <Xm>
        case 0x7a: // CCMP <Wn>, <Wm>, #<nzcv>, <cond>    ;    CCMP <Wn>, #<imm>, #<nzcv>, <cond>     ;    SBCS <Wd>, <Wn>, <Wm>
        case 0xfa: // CCMP <Xn>, <Xm>, #<nzcv>, <cond>    ;    CCMP <Xn>, #<imm>, #<nzcv>, <cond>     ;    SBCS <Xd>, <Xn>, <Xm>
        {
            uint64_t bits23_21 = opbits( 21, 3 );
            uint64_t bits15_10 = opbits( 10, 6 );
            uint64_t n = opbits( 5, 5 );
            bool xregs = ( 0 != ( 0x80 & hi8 ) );

            if ( 2 == bits23_21 )
            {
                uint64_t o3 = opbit( 4 );
                if ( 0 != o3 )
                    unhandled();

                bool is_ccmn = ( 0 == ( hi8 & 0x40 ) );
                uint64_t cond = opbits( 12, 4 );
                uint64_t nzcv = opbits( 0, 4 );
                char width = xregs ? 'x' : 'w';
                uint64_t o2 = opbits( 10, 2 );
                if ( 0 == o2 ) // register
                {
                    uint64_t m = opbits( 16, 5 );
                    tracer.Trace( "%s %c%llu, %c%llu, #%llu, %s\n", is_ccmn ? "ccmn" : "ccmp", width, n, width, m, nzcv, get_cond( cond ) );
                }
                else if ( 2 == o2 ) // immediate
                {
                    uint64_t imm5 = opbits( 16, 5 ); // unsigned
                    tracer.Trace( "%s %c%llu, #%llx, #%llu, %s\n", is_ccmn ? "ccmn" : "ccmp", width, n, imm5, nzcv, get_cond( cond ) );
                }
                else
                    unhandled();
            }
            else if ( ( 0xfa == hi8 || 0x7a == hi8 ) && 0 == bits23_21 && 0 == bits15_10 ) // SBCS <Xd>, <Xn>, <Xm>
            {
                uint64_t d = opbits( 0, 5 );
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "sbcs %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            }
            else if ( ( 0x3a == hi8 || 0xba == hi8 ) && 0 == bits23_21 && 0 == bits15_10 ) // ADCS <Xd>, <Xn>, <Xm>
            {
                uint64_t d = opbits( 0, 5 );
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "adcs %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            }
            else
                unhandled();
            break;
        }
        case 0x31: // ADDS <Wd>, <Wn|WSP>, #<imm>{, <shift>}  ;    CMN <Wn|WSP>, #<imm>{, <shift>}
        case 0xb1: // ADDS <Xd>, <Xn|SP>, #<imm>{, <shift>}   ;    CMN <Xn|SP>, #<imm>{, <shift>}
        {
            uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
            bool shift12 = opbit( 22 );
            uint64_t imm12 = opbits( 10, 12 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );

            if ( 31 == d )
                tracer.Trace( "cmn %s, #%#llx, lsl #%llu\n", reg_or_sp( n, xregs ), imm12, ( shift12 ? 12ull : 0ull ) );
            else
                tracer.Trace( "adds %s, %s, #%#llx, lsl #%llu\n", reg_or_zr( d, xregs ), reg_or_sp( n, xregs ), imm12, ( shift12 ? 12ull : 0ull ) );
            break;
        }
        case 0x0b: // ADD <Wd|WSP>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}      ;    ADD <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
        case 0x2b: // ADDS <Wd>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}         ;    ADDS <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
        case 0x4b: // SUB <Wd|WSP>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}      ;    SUB <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
        case 0x6b: // SUBS <Wd>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}         ;    SUBS <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
        case 0x8b: // ADD <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}      ;    ADD <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
        case 0xab: // ADDS <Xd>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}        ;    ADDS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
        case 0xcb: // SUB <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}      ;    SUB <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
        case 0xeb: // SUBS <Xd>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}        ;    SUBS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
        {
            uint64_t extended = opbit( 21 );
            uint64_t issub = ( 0 != ( 0x40 & hi8 ) );
            const char * opname = issub ? "sub" : "add";
            uint64_t setflags = ( 0 != ( 0x20 & hi8 ) );
            uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
            uint64_t m = opbits( 16, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );

            if ( 1 == extended ) // ADD <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
            {
                uint64_t option = opbits( 13, 3 );
                uint64_t imm3 = opbits( 10, 3 );
                tracer.Trace( "%s%s, %s, %s, %s, %s #%llu\n", opname, setflags ? "s" : "",
                              setflags ? reg_or_zr( d, xregs ) : reg_or_sp( d, xregs ),
                              reg_or_sp( n, xregs ), reg_or_zr( m, xregs ),
                              extend_type( option ), imm3 );
            }
            else // ADD <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
            {
                uint64_t shift = opbits( 22, 2 );
                uint64_t imm6 = opbits( 10, 6 );
                if ( issub && ( 31 == d ) )
                    tracer.Trace( "cmp %s, %s { %s #%llu }\n",
                                  reg_or_zr( n, xregs ), reg_or_zr( m, xregs ),
                                  shift_type( shift ), imm6 );
                else
                    tracer.Trace( "%s%s %s, %s, %s { %s #%llu }\n", opname, setflags ? "s" : "",
                                  reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ),
                                  shift_type( shift ), imm6 );
            }
            break;
        }
        case 0x11: // add <wd|SP>, <wn|SP>, #imm [,<shift>]
        case 0x51: // sub <wd|SP>, <wn|SP>, #imm [,<shift>]
        case 0x91: // add <xd|SP>, <xn|SP>, #imm [,<shift>]
        case 0xd1: // sub <xd|SP>, <xn|SP>, #imm [,<shift>]
        {
            bool sf = ( 0 != opbit( 31 ) );
            bool sh = ( 0 != opbit( 22 ) );
            uint64_t imm12 = opbits( 10, 12 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            tracer.Trace( "%s %s, %s, #%#llx, lsl #%llu\n", ( 0x91 == hi8 || 0x11 == hi8 ) ? "add" : "sub",
                          reg_or_sp( d, sf ), reg_or_sp( n, sf ), imm12, (uint64_t) ( sh ? 12 : 0 ) );
            break;
        }
        case 0xd5: // MSR / MRS
        {
            uint64_t bits2322 = opbits( 22, 2 );
            if ( 0 != bits2322 )
                unhandled();

            if ( 0xd503201f == op ) // nop
            {
                tracer.Trace( "nop\n" );
                break;
            }

            uint64_t upper20 = opbits( 12, 20 );
            uint64_t lower8 = opbits( 0, 8 );
            if ( ( 0xd5033 == upper20 ) && ( 0xbf == lower8 ) ) // dmb -- no memory barries are needed due to just one thread and core
            {
                tracer.Trace( "dmb\n" );
                break;
            }

            uint64_t l = opbit( 21 );
            uint64_t op0 = opbits( 19, 2 );
            uint64_t op1 = opbits( 16, 3 );
            uint64_t op2 = opbits( 5, 3 );
            uint64_t n = opbits( 12, 4 );
            uint64_t m = opbits( 8, 4 );
            uint64_t t = opbits( 0, 5 );

            if ( l ) // MRS <Xt>, (<systemreg>|S<op0>_<op1>_<Cn>_<Cm>_<op2>).   read system register
            {
                if ( ( 3 == op0 ) && ( 14 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 2 == op2 ) ) // cntvct_el0 counter-timer virtual count register
                    tracer.Trace( "mrs x%llu, cntvct_el0\n", t );
                else if ( ( 3 == op0 ) && ( 14 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 0 == op2 ) ) // cntfrq_el0 counter-timer frequency register
                    tracer.Trace( "mrs x%llu, cntfrq_el0\n", t );
                else if ( ( 3 == op0 ) && ( 0 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 7 == op2 ) )
                    tracer.Trace( "mrs x%llu, dczid_elo\n", t );
                else if ( ( 3 == op0 ) && ( 0 == n ) && ( 0 == op1 ) && ( 0 == m ) && ( 0 == op2 ) ) // mrs x, midr_el1
                    tracer.Trace( "mrs x%llu, midr_el1\n", t );
                else if ( ( 3 == op0 ) && ( 13 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 2 == op2 ) )
                    tracer.Trace( "mrs x%llu, tpidr_el0\n", t );
                else if ( ( 3 == op0 ) && ( 4 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 0 == op2 ) ) // mrs x, fpcr
                    tracer.Trace( "mrs x%llu, fpcr // %s\n", t, get_rmode_text( get_bits( fpcr, 22, 2 ) ) );
                else if ( ( 3 == op0 ) && ( 4 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 1 == op2 ) ) // mrs x, fpsr
                    tracer.Trace( "mrs x%llu, fpsr\n", t );
                else
                {
                    tracer.Trace( "MRS unhandled: t %llu op0 %llu n %llu op1 %llu m %llu op2 %llu\n", t, op0, n, op1, m, op2 );
                    unhandled();
                }
            }
            else // MSR.   write system register
            {
                if ( ( 3 == op0 ) && ( 13 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 2 == op2 ) )
                    tracer.Trace( "msr tpidr_el0, x%llu\n", t );
                else if ( ( 0 == op0 ) && ( 2 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 2 == op2 ) )
                    tracer.Trace( "bti\n" ); // branch target identification (ignore );
                else if ( ( 1 == op0 ) && ( 7 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 1 == op2 ) )
                    tracer.Trace( "dc zva, %s\n", reg_or_zr( t, true ) ); // data cache operation
                else if ( ( 0 == op0 ) && ( 2 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 7 == op2 ) ) // xpaclri
                    tracer.Trace( "xpaclri\n" );
                else if ( ( 3 == op0 ) && ( 4 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 0 == op2 ) ) // msr fpcr, xt
                    tracer.Trace( "msr fpcr, x%llu // %s\n", t, get_rmode_text( get_bits( regs[ t ], 22, 2 ) ) );
                else
                {
                    tracer.Trace( "MSR unhandled: t %llu op0 %llu n %llu op1 %llu m %llu op2 %llu\n", t, op0, n, op1, m, op2 );
                    unhandled();
                }
            }
            break;
        }
        case 0x1b: // MADD <Wd>, <Wn>, <Wm>, <Wa>    ;    MSUB <Wd>, <Wn>, <Wm>, <Wa>
        case 0x9b: // MADD <Xd>, <Xn>, <Xm>, <Xa>    ;    MSUB <Xd>, <Xn>, <Xm>, <Xa>    ;    UMULH <Xd>, <Xn>, <Xm>    ;    UMADDL <Xd>, <Wn>, <Wm>, <Xa>
                   // SMADDL <Xd>, <Wn>, <Wm>, <Xa>  ;    SMULH <Xd>, <Xn>, <Xm>         ;    UMSUBL <Xd>, <Wn>, <Wm>, <Xa>
        {
            bool xregs = ( 0 != opbit( 31 ) );
            uint64_t m = opbits( 16, 5 );
            uint64_t a = opbits( 10, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            uint64_t bits23_21 = opbits( 21, 3 );
            bool bit15 = ( 1 == opbit( 15 ) );

            if ( 0x9b == hi8 && 5 == bits23_21 && bit15 ) // UMSUBL <Xd>, <Wn>, <Wm>, <Xa>
                tracer.Trace( "umsubl x%llu, w%llu, w%llu, x%llu\n", d, n, m, a );
            else if ( 1 == bits23_21 && bit15 ) // smsubl
                tracer.Trace( "mmsubl %s, %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ), reg_or_zr( a, xregs ) );
            else if ( 5 == bits23_21 && !bit15 )
                tracer.Trace( "umaddl %s, %s, %s\n", reg_or_zr( d, true ), reg_or_zr( n, true ), reg_or_zr( m, true ) );
            else if ( 1 == bits23_21 && !bit15 )
                tracer.Trace( "smaddl %s, %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, false ), reg_or_zr( m, false ), reg_or_zr( a, xregs ) );
            else if ( 0 == bits23_21 && !bit15 )
                tracer.Trace( "madd %s, %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ), reg_or_zr( a, xregs ) );
            else if ( 0 == bits23_21 && bit15 )
                tracer.Trace( "msub %s, %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ), reg_or_zr( a, xregs ) );
            else if ( 6 == bits23_21 && !bit15 && 31 == a )
                tracer.Trace( "umulh %s, %s, %s\n", reg_or_zr( d, true ), reg_or_zr( n, true ), reg_or_zr( m, true ) );
            else if ( 2 == bits23_21 && !bit15 && 31 == a )
                tracer.Trace( "smulh %s, %s, %s\n", reg_or_zr( d, true ), reg_or_zr( n, true ), reg_or_zr( m, true ) );
            else
                unhandled();
            break;
        }
        case 0x71: // SUBS <Wd>, <Wn|WSP>, #<imm>{, <shift>}   ;   CMP <Wn|WSP>, #<imm>{, <shift>}
        case 0xf1: // SUBS <Xd>, <Xn|SP>, #<imm>{, <shift>}    ;   cmp <xn|SP>, #imm [,<shift>]    ;
        {
            bool sf = ( 0 != opbit( 31 ) );
            bool sh = ( 0 != opbit( 22 ) );
            uint64_t imm12 = opbits( 10, 12 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );

            if ( 31 == d )
                tracer.Trace( "cmp %s, #%#llx, LSL #%llu\n", reg_or_zr( n, sf ), imm12, sh ? 12ull : 0ull );
            else
                tracer.Trace( "subs %s, %s, #%#llx, LSL #%llu\n", reg_or_zr( d, sf ), reg_or_sp( n, sf ), imm12, sh ? 12ull : 0ull );
            break;
        }
        case 0x94: case 0x95: case 0x96: case 0x97: // bl offset. The lower 2 bits of this are the high part of the offset
        {
            int64_t offset = ( opbits( 0, 26 ) << 2 );
            offset = sign_extend( offset, 27 );
            tracer.Trace( "bl %#llx\n", pc + offset );
            break;
        }
        case 0x28: // ldp/stp 32 post index                   STP <Wt1>, <Wt2>, [<Xn|SP>], #<imm>     ;    LDP <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
        case 0xa8: // ldp/stp 64 post-index                   STP <Xt1>, <Xt2>, [<Xn|SP>], #<imm>     ;    LDP <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
        case 0x29: // ldp/stp 32 pre-index and signed offset: STP <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!    ;    STP <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
                   //                                         LDP <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!    ;    LDP <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
        case 0xa9: // ldp/stp 64 pre-index and signed offset: STP <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!    ;    STP <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
                   //                                         LDP <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!    ;    LDP <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
        case 0x68: // ldp 32-bit sign extended                LDPSW <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
        case 0x69: // ldp 32-bit sign extended                LDPSW <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!  ;    LDPSW <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
        {
            bool xregs = ( 0 != opbit( 31 ) );
            uint64_t t1 = opbits( 0, 5 );
            uint64_t t2 = opbits( 10, 5 );
            uint64_t n = opbits( 5, 5 );
            int64_t imm7 = sign_extend( opbits( 15, 7 ), 6 ) << ( xregs ? 3 : 2 );
            uint64_t variant = opbits( 23, 2 );
            if ( 0 == variant )
                unhandled();

            bool postIndex = ( 1 == variant );
            bool preIndex = ( 3 == variant );
            bool signedOffset = ( 2 == variant );

            if ( 0 == opbit( 22 ) ) // bit 22 is 0 for stp
            {
                if ( 0x68 == hi8 || 0x69 == hi8 ) // these are ldpsw variants
                    unhandled();

                if ( signedOffset )
                    tracer.Trace( "stp %s, %s, [%s, #%lld] //so\n", reg_or_zr( t1, xregs ), reg_or_zr( t2, xregs ), reg_or_sp( n, true ), imm7 );
                else if ( preIndex )
                    tracer.Trace( "stp %s, %s, [%s, #%lld]! //pr\n", reg_or_zr( t1, xregs ), reg_or_zr( t2, xregs ), reg_or_sp( n, true ), imm7 );
                else if ( postIndex )
                    tracer.Trace( "stp %s, %s, [%s] #%lld //po\n", reg_or_zr( t1, xregs ), reg_or_zr( t2, xregs ), reg_or_sp( n, true ), imm7 );
                else
                    unhandled();
            }
            else // 1 means ldp
            {
                bool se = ( 0 != ( hi8 & 0x40 ) );
                if ( signedOffset )
                    tracer.Trace( "ldp%s %s, %s, [%s, #%lld] //so\n", se ? "sw" : "", reg_or_zr( t1, xregs ), reg_or_zr( t2, xregs ), reg_or_sp( n, true ), imm7 );
                else if ( preIndex )
                    tracer.Trace( "ldp%s %s, %s, [%s, #%lld]! //pr\n", se ? "sw" : "", reg_or_zr( t1, xregs ), reg_or_zr( t2, xregs ), reg_or_sp( n, true ), imm7 );
                else if ( postIndex )
                    tracer.Trace( "ldp%s %s, %s, [%s] #%lld //po\n", se ? "sw" : "", reg_or_zr( t1, xregs ), reg_or_zr( t2, xregs ), reg_or_sp( n, true ), imm7 );
                else
                    unhandled();
            }
            break;
        }
        case 0x4a: // EOR <Wd>, <Wn>, <Wm>{, <shift> #<amount>}    ;    EON <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
        case 0xca: // EOR <Xd>, <Xn>, <Xm>{, <shift> #<amount>}    ;    EON <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
        case 0x2a: // ORR <Wd>, <Wn>, <Wm>{, <shift> #<amount>}    ;    ORN <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
        case 0xaa: // ORR <Xd>, <Xn>, <Xm>{, <shift> #<amount>}    ;    ORN <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
        {
            uint64_t shift = opbits( 22, 2 );
            uint64_t N = opbit( 21 );
            uint64_t m = opbits( 16, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            uint64_t imm6 = opbits( 10, 6 );
            uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
            bool eor = ( 2 == opbits( 29, 2 ) ); // or eon

            if ( ( 0 == imm6 ) && ( 31 == n ) && ( 0 == shift ) && ( 0 == N ) )
                tracer.Trace( "mov %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( m, xregs ) );
            else if ( ( 0 == shift ) && ( 0 == imm6 ) )
                tracer.Trace( "%s %s, %s, %s\n", eor ? ( N ? "eon" : "eor" ) : ( !N ) ? "orr" : "orn", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            else
                tracer.Trace( "%s %s, %s, %s, %s #%llu\n", eor ? ( N ? "eon" : "eor" ) : ( !N ) ? "orr" : "orn", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ), shift_type( shift ), imm6 );
            break;
        }
        case 0x32: // ORR <Wd|WSP>, <Wn>, #<imm>
        case 0xb2: // ORR <Xd|SP>, <Xn>, #<imm>
        {
            uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
            uint64_t N_immr_imms = opbits( 10, 13 );
            uint64_t op2 = decode_logical_immediate( N_immr_imms, xregs ? 64 : 32 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            tracer.Trace( "orr %s, %s, #%#llx\n", reg_or_sp( d, xregs ), reg_or_zr( n, xregs ), op2 );
            break;
        }
        case 0x33: // BFM <Wd>, <Wn>, #<immr>, #<imms>
        case 0xb3: // BFM <Xd>, <Xn>, #<immr>, #<imms>
        case 0x13: // SBFM <Wd>, <Wn>, #<immr>, #<imms>    ;    EXTR <Wd>, <Wn>, <Wm>, #<lsb>
        case 0x93: // SBFM <Xd>, <Xn>, #<immr>, #<imms>    ;    EXTR <Xd>, <Xn>, <Xm>, #<lsb>
        case 0x53: // UBFM <Wd>, <Wn>, #<immr>, #<imms>
        case 0xd3: // UBFM <Xd>, <Xn>, #<immr>, #<imms>
        {
            uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
            uint64_t imms = opbits( 10, 6 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            uint64_t bit23 = opbit( 23 );
            if ( bit23 && ( 0x13 == ( 0x7f & hi8 ) ) )
            {
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "extr %s, %s, %s, #%llu\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ), imms );
            }
            else
            {
                uint64_t immr = opbits( 16, 6 );
                const char * ins = ( 0x13 == hi8 || 0x93 == hi8 ) ? "sbfm" : ( 0x33 == hi8 || 0xb3 == hi8) ? "bfm" : "ubfm";
                tracer.Trace( "%s %s, %s, #%llu, #%llu\n", ins, reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), immr, imms );
            }
            break;
        }
        case 0x0a: // AND <Wd>, <Wn>, <Wm>{, <shift> #<amount>}     ;    BIC <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
        case 0x6a: // ANDS <Wd>, <Wn>, <Wm>{, <shift> #<amount>}    ;    BICS <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
        case 0x8a: // AND <Xd>, <Xn>, <Xm>{, <shift> #<amount>}     ;    BIC <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
        case 0xea: // ANDS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}    ;    BICS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
        {
            uint64_t shift = opbits( 22, 2 );
            uint64_t N = opbit( 21 );
            uint64_t m = opbits( 16, 5 );
            uint64_t imm6 = opbits( 10, 6 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            bool set_flags = ( 0x60 == ( hi8 & 0x60 ) );
            bool xregs = ( 0 != ( hi8 & 0x80 ) );
            tracer.Trace( "%s%s %s, %s, %s, %s, #%llu\n", N ? "bic" : "and", set_flags ? "s" : "",
                          reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ), shift_type( shift ), imm6 );
            break;
        }
        case 0x10: case 0x30: case 0x50: case 0x70: // ADR <Xd>, <label>
        {
            uint64_t d = opbits( 0, 5 );
            uint64_t immhi = opbits( 5, 19 );
            uint64_t immlo = opbits( 29, 2 );
            int64_t offset = sign_extend( immhi << 2 | immlo, 20 );
            tracer.Trace( "adr x%llu, %#llx\n", d, pc + offset );
            break;
        }
        case 0x90: case 0xb0: case 0xd0: case 0xf0: // adrp rd, immediate
        {
            uint64_t d = ( op & 0x1f );
            int64_t imm = ( ( op >> 3 ) & 0x1ffffc );  // 19 bits with bottom two bits 0 at the bottom
            imm |= opbits( 29, 2 );               // two low bits
            imm = sign_extend( imm, 20 );
            imm <<= 12;
            imm += ( pc & ( ~0xfff ) );
            tracer.Trace( "adrp x%llu, %#llx\n", d, imm );
            break;
        }
        case 0x36: // TBZ <R><t>, #<imm>, <label>
        case 0x37: // TBNZ <R><t>, #<imm>, <label>
        case 0xb6: // TBZ <R><t>, #<imm>, <label> where high bit is prepended to b40 bit selector for 6 bits total
        case 0xb7: // TBNZ <R><t>, #<imm>, <label> where high bit is prepended to b40 bit selector for 6 bits total
        {
            uint64_t b40 = opbits( 19, 5 );
            if ( 0 != ( 0x80 & hi8 ) )
                b40 |= 0x20;
            int64_t imm14 = (int64_t) sign_extend( ( opbits( 5, 14 ) << 2 ), 15 ) + pc;
            uint64_t t = opbits( 0, 5 );
            tracer.Trace( "tb%sz x%llu, #%llu, %#llx\n", ( hi8 & 1 ) ? "n" : "", t, b40, imm14 );
            break;
        }
        case 0x12: // MOVN <Wd>, #<imm>{, LSL #<shift>}   ;    AND <Wd|WSP>, <Wn>, #<imm>
        case 0x92: // MOVN <Xd>, #<imm16>, LSL #<shift>   ;    AND <Xd|SP>, <Xn>, #<imm>    ;    MOV <Xd>, #<imm>
        {
            uint64_t bit23 = opbit( 23 );
            bool xregs = ( 0 != ( hi8 & 0x80 ) );
            if ( bit23 ) // MOVN
            {
                uint64_t d = opbits( 0, 5 );
                uint64_t imm16 = opbits( 5, 16 );
                uint64_t hw = opbits( 21, 2 );
                hw *= 16;
                imm16 <<= hw;
                imm16 = ~imm16;
                char width = 'x';

                if ( 0x12 == hi8 )
                {
                    if ( hw > 16 )
                        unhandled();
                    imm16 = (uint32_t) imm16;
                    width = 'w';
                }

                tracer.Trace( "movn %c%llu, %lld\n", width, d, imm16 );
            }
            else // AND
            {
                uint64_t N_immr_imms = opbits( 10, 13 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t op2 = decode_logical_immediate( N_immr_imms, xregs ? 64 : 32 );
                tracer.Trace( "and %s, %s, #%#llx\n", reg_or_sp( d, xregs ), reg_or_zr( n, xregs ), op2 );
            }
            break;
        }
        case 0x1a: // CSEL <Wd>, <Wn>, <Wm>, <cond>    ;    SDIV <Wd>, <Wn>, <Wm>    ;    UDIV <Wd>, <Wn>, <Wm>    ;    CSINC <Wd>, <Wn>, <Wm>, <cond>
                   // LSRV <Wd>, <Wn>, <Wm>            ;    LSLV <Wd>, <Wn>, <Wm>    ;    ADC <Wd>, <Wn>, <Wm>     ;    ASRV <Wd>, <Wn>, <Wm>
                   // RORV <Wd>, <Wn>, <Wm>
        case 0x9a: // CSEL <Xd>, <Xn>, <Xm>, <cond>    ;    SDIV <Xd>, <Xn>, <Xm>    ;    UDIV <Xd>, <Xn>, <Xm>    ;    CSINC <Xd>, <Xn>, <Xm>, <cond>
                   // LSRV <Xd>, <Xn>, <Xm>            ;    LSLV <Xd>, <Xn>, <Xm>    ;    ADC <Xd>, <Xn>, <Xm>     ;    ASRV <Xd>, <Xn>, <Xm>
                   // RORV <Xd>, <Xn>, <Xm>
        {
            bool xregs = ( 0 != ( hi8 & 0x80 ) );
            uint64_t bits11_10 = opbits( 10, 2 );
            uint64_t d = opbits( 0, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t m = opbits( 16, 5 );
            uint64_t bits15_12 = opbits( 12, 4 );
            uint64_t bits23_21 = opbits( 21, 3 );

            if ( 0 == bits11_10 && 4 == bits23_21 ) // CSEL
            {
                uint64_t cond = opbits( 12, 4 );
                tracer.Trace( "csel %s, %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ), get_cond( cond ) );
            }
            else if ( 1 == bits11_10 && 4 == bits23_21 ) // CSINC <Xd>, XZR, XZR, <cond>
            {
                uint64_t cond = opbits( 12, 4 );
                tracer.Trace( "csinc %s, %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ), get_cond( cond ) );
            }
            else if ( 2 == bits11_10 && 6 == bits23_21 && 2 == bits15_12 ) // ASRV <Xd>, <Xn>, <Xm>
                tracer.Trace( "asrv %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            else if ( 2 == bits11_10 && 6 == bits23_21 && 0 == bits15_12 ) // UDIV <Xd>, <Xn>, <Xm>
                tracer.Trace( "udiv %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            else if ( 3 == bits11_10 && 6 == bits23_21 && 0 == bits15_12 ) // SDIV <Xd>, <Xn>, <Xm>
                tracer.Trace( "sdiv %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            else if ( 1 == bits11_10 && 6 == bits23_21 && 2 == bits15_12 ) // lsrv
                tracer.Trace( "lsrv %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            else if ( 0 == bits11_10 && 6 == bits23_21 && 2 == bits15_12 ) // lslv
                tracer.Trace( "lslv %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            else if ( 0 == bits11_10 && 0 == bits23_21 && 0 == bits15_12 && 0 == bits11_10 ) // addc
                tracer.Trace( "addc %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            else if ( 3 == bits11_10 && 6 == bits23_21 && 2 == bits15_12 ) // RORV <Xd>, <Xn>, <Xm>
                tracer.Trace( "rorv %s, %s, %s\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), reg_or_zr( m, xregs ) );
            else
                unhandled();
            break;
        }
        case 0x52: // MOVZ <Wd>, #<imm>{, LSL #<shift>}    ;    EOR <Wd|WSP>, <Wn>, #<imm>
        case 0xd2: // MOVZ <Xd>, #<imm>{, LSL #<shift>}    ;    EOR <Xd|SP>, <Xn>, #<imm>
        {
            bool xregs = ( 0 != ( hi8 & 0x80 ) );
            uint64_t bit23 = opbit( 23 );

            if ( bit23 ) // movz xd, imm16
            {
                uint64_t d = opbits( 0, 5 );
                uint64_t imm16 = opbits( 5, 16 );
                uint64_t hw = opbits( 21, 2 );
                tracer.Trace( "movz %s, %#llx, LSL #%llu\n", reg_or_zr( d, xregs ), imm16, hw * 16 );
            }
            else // EOR
            {
                uint64_t N_immr_imms = opbits( 10, 13 );
                uint64_t op2 = decode_logical_immediate( N_immr_imms, xregs ? 64 : 32 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                tracer.Trace( "eor %s, %s, #%#llx\n", reg_or_sp( d, xregs ), reg_or_sp( n, xregs ), op2 );
            }
            break;
        }
        case 0x34: case 0xb4: // CBZ <Xt>, <label>     ;    CBZ <WXt>, <label>
        case 0x35: case 0xb5: // CBNZ <Xt>, <label>    ;    CBNZ <Wt>, <label>
        {
            bool xregs = ( 0 != ( hi8 & 0x80 ) );
            uint64_t t = opbits( 0, 5 );
            bool zero_check = ( 0 == ( hi8 & 1 ) );
            int64_t imm19 = ( ( op >> 3 ) & 0x1ffffc ); // two low bits are 0
            imm19 = sign_extend( imm19, 20 );
            tracer.Trace( "cb%sz %s, %#llx\n", zero_check ? "" : "n", reg_or_zr( t, xregs ), pc + imm19 );
            break;
        }
        case 0xd4: // svc
        {
            uint64_t bit23 = opbit( 23 );
            uint64_t hw = opbits( 21, 2 );

            if ( !bit23 && ( 0 == hw ) )
            {
                uint64_t imm16 = opbits( 5, 16 );
                uint64_t op2 = opbits( 2, 3 );
                uint64_t ll = opbits( 0, 2 );
                if ( ( 0 == op2 ) && ( 1 == ll ) ) // svc imm16 supervisor call
                    tracer.Trace( "svc %#llx\n", imm16 );
                else
                    unhandled();
            }
            break;
        }
        case 0x2e: case 0x6e: // CMEQ <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    CMHS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMAXP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // BIT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>     ;    UMINP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    BIF <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // EOR <Vd>.<T>, <Vn>.<T>, <Vm>.<T>     ;    SUB <Vd>.<T>, <Vn>.<T>, <Vm>.<T>     ;    UMULL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                              // MLS <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>] ;  BSL <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;    FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // EXT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<index> ;  INS <Vd>.<Ts>[<index1>], <Vn>.<Ts>[<index2>]  ;    UADDLV <V><d>, <Vn>.<T>
                              // USHL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    FADDP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    FNEG <Vd>.<T>, <Vn>.<T>
                              // CMHI <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UADDW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>  ;   FDIV <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // UMAXV <V><d>, <Vn>.<T> ; UMINV <V><d>, <Vn>.<T>    ;    UMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // FMINNMV S<d>, <Vn>.4S  ; FMAXNMV S<d>, <Vn>.4S     ;    MLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>     ;    UCVTF <Vd>.<T>, <Vn>.<T>
                              // NEG <Vd>.<T>, <Vn>.<T> ; EXT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<index>            ;    FCVTZU <Vd>.<T>, <Vn>.<T>
                              // UMLAL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>           ;    USUBW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
                              // UADDL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>           ;    USUBL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                              // FSQRT <Vd>.<T>, <Vn>.<T>             ;    UMLSL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb> ; NOT <Vd>.<T>, <Vn>.<T>
                              // UADDLP <Vd>.<Ta>, <Vn>.<Tb>          ;    UADALP <Vd>.<Ta>, <Vn>.<Tb>          ;    CMLE <Vd>.<T>, <Vn>.<T>, #0
                              // UQXTN{2} <Vd>.<Tb>, <Vn>.<Ta>        ;    FCMGT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    FCMGE <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // CMGE <Vd>.<T>, <Vn>.<T>, #0          ;    REV32 <Vd>.<T>, <Vn>.<T>
        {
            uint64_t Q = opbit( 30 );
            uint64_t m = opbits( 16, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            uint64_t size = opbits( 22, 2 );
            uint64_t bit23 = opbit( 23 );
            uint64_t bit21 = opbit( 21 );
            uint64_t bit15 = opbit( 15 );
            uint64_t bit10 = opbit( 10 );
            uint64_t bits23_21 = opbits( 21, 3 );
            const char * pT = get_ld1_vector_T( size, Q );
            uint64_t opcode = opbits( 10, 6 );
            uint64_t opcode7 = opbits( 10, 7 );
            uint64_t bits20_17 = opbits( 17, 4 );
            uint64_t bits20_16 = opbits( 16, 5 );
            uint64_t bits16_10 = opbits( 10, 7 );
            uint64_t bits15_10 = opbits( 10, 6 );

            if ( bit21 && 0 == bits20_16 && 2 == bits15_10 ) // REV32 <Vd>.<T>, <Vn>.<T>
            {
                if ( size > 1 )
                    unhandled();
                tracer.Trace( "rev32 v%llu.%s, v%llu.%s\n", d, pT, n, pT );
            }
            else if ( bit21 && 0 == bits20_17 && 0x22 == bits16_10 ) // CMGE <Vd>.<T>, <Vn>.<T>, #0
                tracer.Trace( "cmge v%llu.%s, v%llu.%s, #0\n", d, pT, n, pT );
            else if ( 1 == bits23_21 && 0 == bits20_16 && 0x16 == bits15_10 ) // NOT <Vd>.<T>, <Vn>.<T>. AKA MVN
            {
                pT = Q ? "16b" : "8b";
                tracer.Trace( "not v%llu.%s, v%llu.%s # aka mvn\n", d, pT, n, pT );
            }
            else if ( !bit23 && bit21 && 0x39 == bits15_10 ) // FCMGE <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t sz = opbit( 22 );
                pT = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "fcmge v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit23 && bit21 && 0x39 == bits15_10 ) // FCMGT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t sz = opbit( 22 );
                pT = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "fcmgt v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && 1 == bits20_16 && 0x12 == bits15_10 ) // UQXTN{2} <Vd>.<Tb>, <Vn>.<Ta>
            {
                const char * pTA = get_saddlp_vector_T( size, 1 );
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "uqxtn%s v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTB, n, pTA );
            }
            else if ( bit21 && 0 == bits20_17 && 0x26 == bits16_10 ) // CMLE <Vd>.<T>, <Vn>.<T>, #0
                tracer.Trace( "cmle v%llu.%s, v%llu.%s, #0\n", d, pT, n, pT );
            else if ( bit21 && 0 == bits20_17 && 0x1a == bits16_10 ) // UADALP <Vd>.<Ta>, <Vn>.<Tb>
            {
                const char * pTA = get_saddlp_vector_T( size, Q );
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "uadalp v%llu.%s, v%llu.%s\n", d, pTA, n, pTB );
            }
            else if ( bit21 && 0 == bits20_17 && 0xa == bits16_10 ) // UADDLP <Vd>.<Ta>, <Vn>.<Tb>
            {
                const char * pTA = get_saddlp_vector_T( size, Q );
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "uaddlp v%llu.%s, v%llu.%s\n", d, pTA, n, pTB );
            }
            else if ( bit23 && bit21 && 0 == bits20_17 && 0x7e == bits16_10 ) // FSQRT <Vd>.<T>, <Vn>.<T>
            {
                uint64_t sz = opbit( 22 );
                const char * pTA = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "fsqrt v%llu.%s, v%llu.%s\n", d, pTA, n, pTA );
            }
            else if ( bit21 && 8 == bits15_10 ) // USUBL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
            {
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "reserved";
                tracer.Trace( "uasub%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pT, m, pT );
            }
            else if ( bit21 && 0 == bits15_10 ) // UADDL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
            {
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "reserved";
                tracer.Trace( "uaddl%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pT, m, pT );
            }
            else if ( bit21 && 0xc == bits15_10 ) // USUBW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
            {
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "reserved";
                tracer.Trace( "usubw%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pTA, m, pT );
            }
            else if ( bit21 && 0x20 == bits15_10 ) // UMLAL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
            {
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "reserved";
                tracer.Trace( "umlal%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pT, m, pT );
            }
            else if ( bit21 && 0x28 == bits15_10 ) // UMLSL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
            {
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "reserved";
                tracer.Trace( "umlsl%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pT, m, pT );
            }
            else if ( !bit23 && bit21 && 0 == bits20_17 && 0x76 == bits16_10 ) // UCVTF <Vd>.<T>, <Vn>.<T>
            {
                uint64_t sz = opbit( 22 );
                const char * pTA = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "ucvtf v%llu.%s, v%llu.%s\n", d, pTA, n, pTA );
            }
            else if ( bit23 && bit21 && 0 == bits20_17 && 0x6e == bits16_10 ) // FCVTZU <Vd>.<T>, <Vn>.<T>
            {
                uint64_t sz = opbit( 22 );
                const char * pTA = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "fcvtzu v%llu.%s, v%llu.%s\n", d, pTA, n, pTA );
            }
            else if ( 0 == bits23_21 && !bit15 && !bit10 ) // EXT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<index>
            {
                uint64_t imm4 = opbits( 11, 4 );
                const char * pTA = Q ? "16b" : "8b";
                tracer.Trace( "ext v%llu.%s, v%llu.%s, v%llu.%s, #%llu\n", d, pTA, n, pTA, m, pTA, imm4 );
            }
            else if ( bit21 && 0 == bits20_17 && 0x2e == bits16_10 ) // NEG <Vd>.<T>, <Vn>.<T>
                tracer.Trace( "neg v%llu.%s, v%llu.%s\n", d, pT, n, pT );
            else if ( bit21 && 0x25 == bits15_10 ) // MLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                tracer.Trace( "mls v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            else if ( 0x6e == hi8 && ( 5 == bits23_21 || 1 == bits23_21 ) && 8 == bits20_17 && 0x32 == bits16_10 ) // FMINNMV S<d>, <Vn>.4S    ;    FMAXNMV S<d>, <Vn>.4S
                tracer.Trace( "%s s%llu, v%llu.4s\n", 5 == bits23_21 ? "fminnmv" : "fmaxnmv", d, n );
            else if ( !bit23 && bit21 && 0x3f == bits15_10 ) // FDIV <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t sz = opbit( 22 );
                pT = ( !sz && !Q ) ? "2s" : ( !sz && Q ) ? "4s" : ( sz && Q ) ? "2d" : "?";
                tracer.Trace( "v%llu.%s, v%llu.%s, v%llu.%s\n", n, pT, n, pT, m, pT );
            }
            else if ( bit21 && ( 0x1b == bits15_10 || 0x19 == bits15_10 ) ) // UMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                tracer.Trace( "%s v%llu.%s, v%llu.%s, %llu.%s\n", ( 0x1b == bits15_10 ) ? "umin" : "umax", d, pT, n, pT, m, pT );
            else if ( bit21 && 8 == bits20_17 && ( 0x6a == bits16_10 || 0x2a == bits16_10 ) ) // UMINV <V><d>, <Vn>.<T>    ;    UMAXV <V><d>, <Vn>.<T>
            {
                char v = ( 0 == size ) ? 'b' : ( 1 == size ) ? 'h' : ( 2 == size ) ? 's' : '?';
                tracer.Trace( "%s %c%llu, v%llu.%s\n", ( 0x6a == bits16_10 ) ? "uminv" : "unaxv", v, d, n, pT );
            }
            else if ( bit21 && 4 == bits15_10 ) // UADDW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
            {
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "?";
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "uaddw%s, v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pTA, m, pTB );
            }
            else if ( bit23 && bit21 && 0 == bits20_17 && 0x3e == bits16_10 ) // FNEG <Vd>.<T>, <Vn>.<T>
            {
                uint64_t sz = opbit( 22 );
                uint64_t ty = ( sz << 1 ) | Q;
                pT = ( 0 == ty ) ? "2s" : ( 1 == ty ) ? "4s" : ( 3 == ty ) ? "2d" : "?";
                tracer.Trace( "fneg v%llu.%s, v%llu.%s\n", d, pT, n, pT );
            }
            else if ( !bit23 && bit21 && 0x35 == opcode ) // FADDP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t sz = opbit( 22 );
                uint64_t ty = ( sz << 1 ) | Q;
                pT = ( 0 == ty ) ? "2s" : ( 1 == ty ) ? "4s" : ( 3 == ty ) ? "2d" : "?";
                tracer.Trace( "faddp v%llu.%s, v%llu,%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && 0x11 == opcode ) // USHL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                tracer.Trace( "ushl, v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            else if ( bit21 && 8 == bits20_17 && 0xe == opcode7 ) // UADDLV <V><d>, <Vn>.<T>
                tracer.Trace( "uaddlv v%llu, v%llu.%s\n", d, n, pT );
            else if ( 0x6e == hi8 && 0 == bits23_21 && !bit15 && bit10 ) // INS <Vd>.<Ts>[<index>], <R><n>
            {
                uint64_t imm5 = opbits( 16, 5 );
                uint64_t imm4 = opbits( 11, 5 );
                uint64_t index1 = 0;
                uint64_t index2 = 0;
                char T = '?';
                if ( 1 & imm5 )
                {
                    index1 = get_bits( imm5, 1, 4 );
                    index2 = imm4;
                    T = 'B';
                }
                else if ( 2 & imm5 )
                {
                    index1 = get_bits( imm5, 2, 3 );
                    index2 = get_bits( imm4, 1, 3 );
                    T = 'H';
                }
                else if ( 4 & imm5 )
                {
                    index1 = get_bits( imm5, 3, 2 );
                    index2 = get_bits( imm4, 2, 2 );
                    T = 'S';
                }
                else if ( 8 & imm5 )
                {
                    index1 = get_bit( imm5, 4 );
                    index2 = get_bit( imm4, 3 );
                    T = 'D';
                }

                tracer.Trace( "ins v%llu.%c[%llu], v%llu.%c[%llu]\n", d, T, index1, n, T, index2 );
            }
            else if ( bit21 && 0x0d == opcode )
                tracer.Trace( "cmhi v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            else if ( bit21 && 0x23 == opcode )
                tracer.Trace( "cmeq v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            else if ( bit21 && 0x0f == opcode )
                tracer.Trace( "cmhs v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            else if ( bit21 && 0x29 == opcode )
                tracer.Trace( "umaxp v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            else if ( bit21 && 0x2b == opcode )
                tracer.Trace( "uminp v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            else if ( bit21 && 0x07 == opcode )
            {
                uint64_t opc2 = opbits( 22, 2 );
                pT = ( 0 == Q ) ? "8B" : "16B";
                tracer.Trace( "%s v%llu.%s, v%llu.%s, v%llu.%s\n", ( 1 == opc2 ) ? "bsl" : ( 2 == opc2) ? "bit" : ( 3 == opc2 ) ? "bif" : "eor", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && 0x21 == opcode )
                tracer.Trace( "sub v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            else if ( bit21 && 0x30 == opcode )
            {
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "?";
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "umull%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pTB, m, pTB );
            }
            else if ( bit21 && 0x25 == opcode ) // MLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                if ( 3 == size )
                    unhandled();
                tracer.Trace( "mls v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && 0x37 == opcode )
            {
                uint64_t sz = opbit( 22 );
                pT = ( 0 == sz ) ? ( 0 == Q ? "2S" : "4S" ) : ( 0 == Q ? "?" : "2D" );
                tracer.Trace( "fmul v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( !bit21 && 0 == size && !bit10 && !bit15 )
            {
                uint64_t imm4 = opbits( 11, 4 );
                pT = Q ? "8B" : "16B";
                tracer.Trace( "ext v%llu.%s, v%llu.%s, v%llu.%s, #%llu\n", d, pT, n, pT, m, pT, imm4 );
            }
            else
                unhandled();
            break;
        }
        case 0x5e: // SCVTF <V><d>, <V><n>    ;    ADDP D<d>, <Vn>.2D    ;    DUP <V><d>, <Vn>.<T>[<index>]    ;    FCVTZS <V><d>, <V><n>
                   // CMGT D<d>, D<n>, D<m>   ;    CMGT D<d>, D<n>, #0   ;    ADD D<d>, D<n>, D<m>             ;    FCMLT <V><d>, <V><n>, #0.0
                   // CMEQ D<d>, D<n>, #0
        {
            uint64_t bits23_10 = opbits( 10, 14 );
            uint64_t bit23 = opbit( 23 );
            uint64_t bit21 = opbit( 21 );
            uint64_t bits23_21 = opbits( 21, 3 );
            uint64_t bits20_16 = opbits( 16, 5 );
            uint64_t bits15_10 = opbits( 10, 6 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );

            if ( 7 == bits23_21 && 0 == bits20_16 && 0x26 == bits15_10 ) // CMEQ D<d>, D<n>, #0
                tracer.Trace( "cmeq d%llu, d%llu, #0\n", d, n );
            else if ( bit23 && bit21 && 0 == bits20_16 && 0x3a == bits15_10 ) // FCMLT <V><d>, <V><n>, #0.0
            {
                uint64_t sz = opbit( 22 );
                char width = sz ? 'd' : 's';
                tracer.Trace( "%c%llu, %c%llu, #0.0\n", width, d, width, n );
            }
            else if ( 7 == bits23_21 && 0x21 == bits15_10 ) // ADD D<d>, D<n>, D<m>
            {
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "add d%llu, d%llu, d%llu\n", d, n, m );
            }
            else if ( 7 == bits23_21 && 0xd == bits15_10 ) // CMGT D<d>, D<n>, D<m>
            {
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "cmgt d%llu, d%llu, d%llu\n", d, n, m );
            }
            else if ( 0x3822 == bits23_10 ) // CMGT D<d>, D<n>, #0
            {
                tracer.Trace( "cmgt d%llu, d%llu, #0\n", d, n ); // code not implemented
            }
            else if ( 0x386e == bits23_10 || 0x286e == bits23_10 ) // FCVTZS <V><d>, <V><n>
            {
                uint64_t sz = opbit( 22 );
                char width = sz ? 'd' : 's';
                tracer.Trace( "fcvtzs %c%llu, %c%llu\n", width, d, width, n );
            }
            else if ( 0x0876 == ( bits23_10 & 0x2fff ) ) // SCVTF <V><d>, <V><n>
            {
                uint64_t sz = opbit( 22 );
                char width = sz ? 'd' : 's';
                tracer.Trace( "scvtf %c%llu, %c%llu\n", width, d, width, n );
            }
            else if ( 0x3c6e == bits23_10 ) // DUP <V><d>, <Vn>.<T>[<index>]
                tracer.Trace( "addp D%llu, v%llu.2D\n", d, n );
            else if ( 1 == ( bits23_10 & 0x383f ) ) // DUP <V><d>, <Vn>.<T>[<index>]   -- scalar
            {
                uint64_t imm5 = opbits( 16, 5 );
                uint64_t size = lowest_set_bit_nz( imm5 & 0xf );
                uint64_t index = get_bits( imm5, size + 1, size + 2 ); // imm5:<4:size+1>
                const char * pT = ( imm5 & 1 ) ? "B" : ( imm5 & 2 ) ? "H" : ( imm5 & 4 ) ? "S" : "D";
                tracer.Trace( "dup %s%llu, v%llu.%s[%llu]\n", pT, d, n, pT, index );
            }
            else
                unhandled();
            break;
        }
        case 0x7e: // CMGE    ;    UCVTF <V><d>, <V><n>    ;    UCVTF <Hd>, <Hn>            ;    FADDP <V><d>, <Vn>.<T>    ;    FABD <V><d>, <V><n>, <V><m>
                   // FCMGE <V><d>, <V><n>, #0.0           ;    FMINNMP <V><d>, <Vn>.<T>    ;    FMAXNMP <V><d>, <Vn>.<T>
                   // CMHI D<d>, D<n>, D<m>                ;    FCVTZU <V><d>, <V><n>       ;    FCMGT <V><d>, <V><n>, <V><m>
                   // FCMGE <V><d>, <V><n>, <V><m>         ;    CMLE D<d>, D<n>, #0         ;    CMGE D<d>, D<n>, #0
        {
            uint64_t bits23_10 = opbits( 10, 14 );
            uint64_t bits23_21 = opbits( 21, 3 );
            uint64_t bits20_10 = opbits( 10, 11 );
            uint64_t bits15_10 = opbits( 10, 6 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            uint64_t bit23 = opbit( 23 );
            uint64_t sz = opbit( 22 );
            uint64_t bit21 = opbit( 21 );
            uint64_t opcode = opbits( 10, 6 );

            if ( !bit23 && bit21 && 0x39 == bits15_10 ) // FCMGE <V><d>, <V><n>, <V><m>
            {
                char width = sz ? 'd' : 's';
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "fcmge %c%llu, %c%llu, %c%llu\n", width, d, width, n, width, m );
            }
            else if ( bit23 && bit21 && 0x39 == bits15_10 ) // FCMGT <V><d>, <V><n>, <V><m>
            {
                char width = sz ? 'd' : 's';
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "fcmgt %c%llu, %c%llu, %c%llu\n", width, d, width, n, width, m );
            }
            else if ( bit23 && bit21 && 0x6e == bits20_10 ) // FCVTZU <V><d>, <V><n>
            {
                char width = sz ? 'd' : 's';
                tracer.Trace( "fcvtzu %c%llu, %c%llu\n", width, d, width, n );
            }
            else if ( 7 == bits23_21 && 0xd == bits15_10 ) // CMHI D<d>, D<n>, D<m>
            {
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( " cmhi d%llu, d%llu, d%llu\n", d, n, m );
            }
            else if ( bit21 && 0x432 == bits20_10 ) // FMINNMP <V><d>, <Vn>.<T>    ;    FMAXNMP <V><d>, <Vn>.<T>
            {
                tracer.Trace( "%s %c%llu, v%llu.%s\n", bit23 ? "fminnmp" : "fmaxnmp", sz ? 'd' : 's', d, n, sz ? "2d" : "2s" );
            }
            else if ( bit23 && bit21 && 0x35 == opcode ) // FABD <V><d>, <V><n>, <V><m>
            {
                char width = sz ? 'd' : 's';
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "fabd %c%llu, %c%llu, %c%llu\n", width, d, width, n, width, m );
            }
            else if ( 0x0c36 == bits23_10 || 0x1c36 == bits23_10 ) // FADDP <V><d>, <Vn>.<T>
            {
                char width = sz ? 'd' : 's';
                tracer.Trace( "faddp %c%llu, v%llu.2%c\n", width, d, n, width );
            }
            else if ( 0x3822 == bits23_10 ) // CMGE D<d>, D<n>, #0
                tracer.Trace( "cmge d%llu, d%llu, #0\n", d, n );
            else if ( 0x3826 == bits23_10 ) // CMLE D<d>, D<n>, #0
                tracer.Trace( "cmle d%llu, d%llu, #0\n", d, n );
            else if ( 0x0876 == ( bits23_10 & 0x2fff ) )
            {
                char width = sz ? 'd' : 's';
                tracer.Trace( "ucvtf %c%llu, %c%llu\n", width, d, width, n );
            }
            else if ( 0x2832 == bits23_10 || 0x3832 == bits23_10 ) // FCMGE <V><d>, <V><n>, #0.0
            {
                char type = sz ? 'd' : 'f';
                tracer.Trace( "fcmge %c%llu, %c%llu, #0.0\n", type, d, type, n );
            }
            else
                unhandled();
            break;
        }
        case 0x0e: case 0x4e: // DUP <Vd>.<T>, <Vn>.<Ts>[<index>]    ;    DUP <Vd>.<T>, <R><n>    ;             CMEQ <Vd>.<T>, <Vn>.<T>, #0    ;    ADDP <Vd>.<T>, <V
                              // AND <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMOV <Wd>, <Vn>.<Ts>[<index>]    ;    UMOV <Xd>, <Vn>.D[<index>]     ;    CNT <Vd>.<T>, <Vn>.<T>
                              // AND <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMOV <Wd>, <Vn>.<Ts>[<index>]    ;    UMOV <Xd>, <Vn>.D[<index>]     ;    ADDV <V><d>, <Vn>.<T>
                              // XTN{2} <Vd>.<Tb>, <Vn>.<Ta>         ;    UZP1 <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   UZP2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // SMOV <Wd>, <Vn>.<Ts>[<index>]       ;    SMOV <Xd>, <Vn>.<Ts>[<index>]    ;    INS <Vd>.<Ts>[<index>], <R><n> ;    CMGT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // SCVTF <Vd>.<T>, <Vn>.<T>            ;    FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<T>;    FADD <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // TRN1 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    TRN2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   TBL <Vd>.<Ta>, { <Vn>.16B }, <Vm>.<Ta> ; TBL <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B, <Vn+2>.16B, <Vn+3>.16B }, <Vm>.<Ta>
                              // ZIP1 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    ZIP2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   SMULL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                              // MLA <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    MUL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>  ;   CMLT <Vd>.<T>, <Vn>.<T>, #0    ;    REV64 <Vd>.<T>, <Vn>.<T>
                              // BIC <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    FMLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   FSUB <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // SMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    SMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   SMINV <V><d>, <Vn>.<T>         ;    SMAXV <V><d>, <Vn>.<T>
                              // FMINNM <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;    FMAXNM <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ; FCVTN{2} <Vd>.<Tb>, <Vn>.<Ta>  ;    FCVTZS <Vd>.<T>, <Vn>.<T>
                              // ORN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    FCVTL{2} <Vd>.<Ta>, <Vn>.<Tb>     ;   SSHL <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ; SADDW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
                              // CMGE <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    ABS <Vd>.<T>, <Vn>.<T>            ;   SSUBW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
                              // FCMLT <Vd>.<T>, <Vn>.<T>, #0.0      ;    FABS <Vd>.<T>, <Vn>.<T>           ;   SMLAL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                              // SADDL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb> ; SADDLP <Vd>.<Ta>, <Vn>.<Tb>     ;   SADDLV <V><d>, <Vn>.<T>        ;    SADALP <Vd>.<Ta>, <Vn>.<Tb>
                              // SQXTN{2} <Vd>.<Tb>, <Vn>.<Ta>       ;    CMGT <Vd>.<T>, <Vn>.<T>, #0       ;   CMTST <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                              // FMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    FCMEQ <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;  REV16 <Vd>.<T>, <Vn>.<T>
        {
            uint64_t Q = opbit( 30 );
            uint64_t imm5 = opbits( 16, 5 );
            uint64_t bit15 = opbit( 15 );
            uint64_t bits14_11 = opbits( 11, 4 );
            uint64_t bit10 = opbit( 10 );
            uint64_t bits12_10 = opbits( 10, 3 );
            uint64_t bit21 = opbit( 21 );
            uint64_t bit23 = opbit( 23 );
            uint64_t bits23_21 = opbits( 21, 3 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            uint64_t bits20_16 = opbits( 16, 5 );
            uint64_t bits14_10 = opbits( 10, 5 );
            uint64_t bits15_10 = opbits( 10, 6 );

            if ( bit21 && 0 == bits20_16 && 6 == bits15_10 ) // REV16 <Vd>.<T>, <Vn>.<T>
            {
                uint64_t size = opbits( 22, 2 );
                if ( 0 != size )
                    unhandled();
                const char * pT = Q ? "16b" : "8b";
                tracer.Trace( "rev16 v%llu.%s, v%llu.%s\n", d, pT, n, pT );
            }
            else if ( !bit23 && bit21 && 0x39 == bits15_10 ) // FCMEQ <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t sz = opbit( 22 );
                const char * pT = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "fcmeq v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && 0x3d == bits15_10 ) // FMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    FMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t sz = opbit( 22 );
                const char * pT = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "%s v%llu.%s, v%llu.%s, v%llu.%s\n", bit23 ? "fmin" : "fmax", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && 0x23 == bits15_10 ) // CMTST <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t size = opbits( 22, 2 );
                uint64_t m = opbits( 16, 5 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "cmtst v%llu.%s, v%llu.%s, #0\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && 0 == bits20_16 && 0x22 == bits15_10 ) // CMGT <Vd>.<T>, <Vn>.<T>, #0
            {
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "cmgt v%llu.%s, v%llu.%s, #0\n", d, pT, n, pT );
            }
            else if ( bit21 && 1 == bits20_16 && 0x12 == bits15_10 ) // SQXTN{2} <Vd>.<Tb>, <Vn>.<Ta>
            {
                uint64_t size = opbits( 22, 2 );
                const char * pTA = get_saddlp_vector_T( size, 1 );
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "sqxtn%s v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTB, n, pTA );
            }
            else if ( bit21 && 0 == bits20_16 && 0x1a == bits15_10 ) // SADALP <Vd>.<Ta>, <Vn>.<Tb>
            {
                uint64_t size = opbits( 22, 2 );
                const char * pTA = get_saddlp_vector_T( size, Q );
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "sadalp v%llu.%s, v%llu.%s\n", d, pTA, n, pTB );
            }
            else if ( bit21 && 0x10 == bits20_16 && 0x0e == bits15_10 ) // SADDLV <V><d>, <Vn>.<T>
            {
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                char v = ( 0 == size ) ? 'h' : ( 1 == size ) ? 's' : ( 2 == size ) ? 'd' : '?';
                tracer.Trace( "saddlv %c%llu, v%llu.%s\n", v, d, n, pT );
            }
            else if ( bit21 && 0 == bits20_16 && 0x0a == bits15_10 ) // SADDLP <Vd>.<Ta>, <Vn>.<Tb>
            {
                uint64_t size = opbits( 22, 2 );
                const char * pTA = get_saddlp_vector_T( size, Q );
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "saddlp v%llu.%s, v%llu.%s\n", d, pTA, n, pTB );
            }
            else if ( bit21 && 0 == bits15_10 ) // SADDL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
            {
                uint64_t size = opbits( 22, 2 );
                uint64_t m = opbits( 16, 5 );
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "unknown";
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "saddl%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pTB, m, pTB );
            }
            else if ( bit21 && 0x20 == bits15_10 ) // SMLAL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
            {
                uint64_t size = opbits( 22, 2 );
                uint64_t m = opbits( 16, 5 );
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "unknown";
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "smlal%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pTB, m, pTB );
            }
            else if ( bit23 && bit21 && 0 == bits20_16 && 0x3e == bits15_10 ) // FABS <Vd>.<T>, <Vn>.<T>
            {
                uint64_t sz = opbit( 22 );
                const char * pT = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "fabs v%llu.%s, v%llu.%s, #0.0\n", d, pT, n, pT );
            }
            else if ( bit23 && bit21 && 0 == bits20_16 && 0x3a == bits15_10 ) // FCMLT <Vd>.<T>, <Vn>.<T>, #0.0
            {
                uint64_t sz = opbit( 22 );
                const char * pT = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "fcmlt v%llu.%s, v%llu.%s, #0.0\n", d, pT, n, pT );
            }
            else if ( bit21 && 0xc == bits15_10 ) // SSUBW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "unknown";
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "ssubw%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pTA, m, pTB );
            }
            else if ( bit21 && 0 == bits20_16 && 0x2e == bits15_10 ) // ABS <Vd>.<T>, <Vn>.<T>
            {
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "abs v%llu.%s, v%llu.%s\n", d, pT, n, pT );
            }
            else if ( bit21 && 4 == bits15_10 ) // SADDW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "unknown";
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "saddw%s v%llu.%s, v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pTA, m, pTB );
            }
            else if ( bit21 && 0x11 == bits15_10 ) // SSHL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "sshl v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( !bit23 && bit21 && 1 == bits20_16 && 0x1e == bits15_10 ) // FCVTL{2} <Vd>.<Ta>, <Vn>.<Tb>
            {
                uint64_t sz = opbit( 22 );
                const char * pTA = sz ? "2d" : "4s";
                const char * pTB = sz ? Q ? "4s" : "2s" : Q ? "8h" : "4h";
                tracer.Trace( "fcvtl%s v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTA, n, pTB );
            }
            else if ( 7 == bits23_21 && 7 == bits15_10 ) // ORN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                const char * pT = Q ? "16b" : "8b";
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "orn v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit23 && bit21 && 1 == bits20_16 && 0x2e == bits15_10 ) // FCVTZS <Vd>.<T>, <Vn>.<T>
            {
                uint64_t sz = opbit( 22 );
                const char * pT = sz ? Q ? "2d" : "reserved" : Q ? "4s" : "2s";
                tracer.Trace( "fcvtzs v%llu.%s, v%llu.%s\n", d, pT, n, pT );
            }
            else if ( !bit23 && bit21 && 1 == bits20_16 && 0x1a == bits15_10 ) // FCVTN{2} <Vd>.<Tb>, <Vn>.<Ta>
            {
                uint64_t sz = opbit( 22 );
                const char * pTA = sz ? "2d" : "4s";
                const char * pTB = sz ? Q ? "8h" : "4h" : Q ? "4s" : "2s";
                tracer.Trace( "fcvtn%s v%llu.%s, v%llu.%s  # rmode %s\n", Q ? "2" : "", d, pTB, n, pTA, get_rmode_text( get_bits( fpcr, 22, 2 ) ) );
            }
            else if ( bit21 && 0x31 == bits15_10 ) // FMINNM <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;    FMAXNM <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t sz = opbit( 22 );
                const char * pT = ( !sz && !Q ) ? "2s" : ( !sz && Q ) ? "4s" : ( sz && Q ) ? "2d" : "?";
                tracer.Trace( "%s v%llu,%s, v%llu.%s, v%llu.%s\n", bit23 ? "fminnm" : "fmaxnm", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && ( 0x11 == bits20_16 || 0x10 == bits20_16 ) && 0x2a == bits15_10 ) // SMINV <V><d>, <Vn>.<T>         ;    SMAXV <V><d>, <Vn>.<T>
            {
                uint64_t size = opbits( 22, 2 );
                char v = ( 0 == size ) ? 'b' : ( 1 == size ) ? 'h' : ( 2 == size ) ? 's' : '?';
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "%s %c%llu, v%llu.%s\n", ( 0x11 == bits20_16 ) ? "sminv" : "smaxv", v, d, n, pT );
            }
            else if ( bit21 && ( 0x19 == bits15_10 || 0x1b == bits15_10 ) ) // SMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    SMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "%s v%llu.%s, v%llu.%s, v%llu.%s\n", ( 0x19 == bits15_10 ) ? "smax" : "smin", d, pT, n, pT, m, pT );
            }
            else if ( bit23 && bit21 && 0x35 == bits15_10 ) // FSUB <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t sz = opbit( 22 );
                const char * pT = ( !sz && !Q ) ? "2s" : ( !sz && Q ) ? "4s" : ( sz && Q ) ? "2d" : "?";
                tracer.Trace( "fsub v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit23 && bit21 && 0x33 == bits15_10 ) // FMLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t sz = opbit( 22 );
                const char * pT = ( !sz && !Q ) ? "2s" : ( !sz && Q ) ? "4s" : ( sz && Q ) ? "2d" : "?";
                tracer.Trace( "fmls v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( 3 == bits23_21 && 7 == bits15_10 ) // BIC <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                const char * pT = Q ? "16b" : "8b";
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "bic v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && 0 == bits20_16 && !bit15 && 2 == bits14_10 ) // REV64 <Vd>.<T>, <Vn>.<T>
            {
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "rev64 v%llu.%s, v%llu.%s\n", d, pT, n, pT );
            }
            else if ( bit21 && 0 == bits20_16 && bit15 && 0xa == bits14_10 ) // CMLT <Vd>.<T>, <Vn>.<T>, #0
            {
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "cmlt v%llu.%s, v%llu.%s, #0\n", d, pT, n, pT );
            }
            else if ( bit21 && bit15 && ( 7 == bits14_10 || 5 == bits14_10 ) ) // MUL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    MLA <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "%s v%llu.%s, v%llu.%s, v%llu.%s\n", ( 7 == bits14_10 ) ? "mul" : "mla", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && bit15 && 8 == bits14_11 && !bit10 ) // SMULL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                uint64_t part = Q;
                const char * pTA = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : ( 2 == size ) ? "2d" : "unknown";
                const char * pTB = get_ld1_vector_T( size, Q );
                tracer.Trace( "smull%s v%llu.%s, v%llu.%s, v%llu.%s\n", part ? "2" : "", d, pTA, n, pTB, m, pTB );
            }
            else if ( !bit21 && !bit15 && ( 0x1e == bits14_10 || 0xe == bits14_10 ) ) // ZIP1/2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "zip%c v%llu.%s, v%llu.%s, v%llu.%s\n", ( 0x1e == bits14_10 ) ? '2' : '1', d, pT, n, pT, m, pT );
            }
            else if ( 0 == bits23_21 && !bit15 && 0 == bits12_10 ) // TBL <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B, <Vn+2>.16B, <Vn+3>.16B }, <Vm>.<Ta>
            {
                uint64_t m = opbits( 16, 5 );
                const char * pT = Q ? "16b" : "8b";
                uint64_t len = opbits( 13, 2 );
                if ( 0 == len )
                    tracer.Trace( "tbl v%llu.%s, {v%llu.16b}, v%llu.%s\n", d, pT, n, m, pT );
                else if ( 1 == len )
                    tracer.Trace( "tbl v%llu.%s, {v%llu.16b, v%llu.16b}, v%llu.%s\n", d, pT, n, n + 1, m, pT );
                else if ( 2 == len )
                    tracer.Trace( "tbl v%llu.%s, {v%llu.16b, v%llu.16b, v%llu.16b }, v%llu.%s\n", d, pT, n, n + 1, n + 2, m, pT );
                else if ( 3 == len )
                    tracer.Trace( "tbl v%llu.%s, {v%llu.16b, v%llu.16b, v%llu.16b, v%llu.16b }, v%llu.%s\n", d, pT, n, n + 1, n + 2, n + 3, m, pT );
            }
            else if ( !bit21 && !bit15 && 0xd == bits14_11 && !bit10 ) // TRN2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "trn2 v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( !bit21 && !bit15 && 5 == bits14_11 && !bit10 ) // TRN1 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "trn1 v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( !bit23 && bit21 && bit15 && 0xa == bits14_11 && bit10 ) // FADD <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t sz = opbit( 22 );
                uint64_t ty = ( sz << 1 ) | Q;
                const char * pT = ( 0 == ty ) ? "2s" : ( 1 == ty ) ? "4s" : ( 3 == ty ) ? "2d" : "?";
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "fadd v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( !bit23 && bit21 && bit15 && 9 == bits14_11 && bit10 ) // FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t sz = opbit( 22 );
                uint64_t ty = ( sz << 1 ) | Q;
                const char * pT = ( 0 == ty ) ? "2s" : ( 1 == ty ) ? "4s" : ( 3 == ty ) ? "2d" : "?";
                uint64_t m = opbits( 16, 5 );
                tracer.Trace( "fmla v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( !bit23 && bit21 && 1 == bits20_16 && bit15 && 0x16 == bits14_10 ) // SCVTF <Vd>.<T>, <Vn>.<T>
            {
                uint64_t sz = opbit( 22 );
                uint64_t ty = ( sz << 1 ) | Q;
                const char * pT = ( 0 == ty ) ? "2s" : ( 1 == ty ) ? "4s" : ( 3 == ty ) ? "2d" : "?";
                tracer.Trace( "scvtf v%llu.%s, v%llu.%s\n", d, pT, n, pT );
            }
            else if ( 0x4e == hi8 && 0 == bits23_21 && !bit15 && 3 == bits14_11 && bit10 ) // INS <Vd>.<Ts>[<index>], <R><n>
            {
                char T = '?';
                uint64_t index = 0;
                if ( imm5 & 1 )
                {
                    T = 'B';
                    index = get_bits( imm5, 1, 4 );
                }
                else if ( imm5 & 2 )
                {
                    T = 'H';
                    index = get_bits( imm5, 2, 3 );
                }
                else if ( imm5 & 4 )
                {
                    T = 'S';
                    index = get_bits( imm5, 3, 2 );
                }
                else if ( imm5 & 8 )
                {
                    T = 'D';
                    index = get_bit( imm5, 4 );
                }
                else
                    unhandled();
                tracer.Trace( "ins v%llu.%c[%llu], %s\n", d, T, index, reg_or_zr( n, ( 8 == ( imm5 & 0xf ) ) ) );
            }
            else if ( !bit21 && !bit15 && ( 7 == bits14_11 || 5 == bits14_11 ) && bit10 )
            {
                // UMOV <Wd>, <Vn>.<Ts>[<index>]    ;    UMOV <Xd>, <Vn>.D[<index>]    ;     SMOV <Wd>, <Vn>.<Ts>[<index>]    ;    SMOV <Xd>, <Vn>.<Ts>[<index>]
                uint64_t size = lowest_set_bit_nz( imm5 & ( ( 7 == bits14_11 ) ? 0xf : 7 ) );
                uint64_t bits_to_copy = 4 - size;
                uint64_t index = get_bits( imm5, 4 + 1 - bits_to_copy, bits_to_copy );

                const char * pT = "UNKNOWN";
                if ( imm5 & 1 )
                    pT = "B";
                else if ( imm5 & 2 )
                    pT = "H";
                else if ( imm5 & 4 )
                    pT = "S";
                else if ( imm5 & 8 )
                    pT = "D";
                else
                    unhandled();
                tracer.Trace( "%cmov %s, v%llu.%s[%llu]\n", ( 7 == bits14_11 ) ? 'u' : 's', reg_or_zr( d, Q ), n, pT, index );
            }
            else if ( !bit21 && !bit15 && ( 0x3 == bits14_11 || 0xb == bits14_11 ) && !bit10 ) //
            {
                uint64_t size = opbits( 22, 2 );
                uint64_t part = opbit( 14 );
                uint64_t m = imm5;
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "uzp%c v%llu.%s, v%llu.%s, v%llu.%s\n", ( 1 == part ) ? '2' : '1', d, pT, n, pT, m, pT );
            }
            else if ( 1 == bits23_21 && !bit15 && 3 == bits14_11 && bit10 ) // AND <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = imm5;
                const char * pT = ( 0 == Q ) ? "8B" : "16B";
                tracer.Trace( "and v%llu.%s, v%llu.%s v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( 5 == bits23_21 && !bit15 && 3 == bits14_11 && bit10 ) // ORR <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = imm5;
                const char * pT = ( 0 == Q ) ? "8B" : "16B";
                tracer.Trace( "orr v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && bit15 && 3 == bits14_11 && !bit10 && 0 == bits20_16 )  // CMEQ <Vd>.<T>, <Vn>.<T>, #0
            {
                uint64_t size = opbits( 22, 2 );
                tracer.Trace( "cmeq v%llu.%s, v%llu.%s, #0\n", d, get_ld1_vector_T( size, Q ), n, get_ld1_vector_T( size, Q ) );
            }
            else if ( bit21 && !bit15 && 7 == bits14_11 && bit10 ) // CMGE <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "cmge v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && !bit15 && 6 == bits14_11 && bit10 ) // CMGT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "cmgt v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && bit15 && 7 == bits14_11 && bit10 ) // ADDP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t size = opbits( 22, 2 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "addp v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( 0 == bits23_21 && !bit15 && 1 == bits14_11 && bit10 ) // DUP <Vd>.<T>, <R><n>
                tracer.Trace( "dup v%llu.%s, %s\n", d, get_vector_T( imm5, Q ), reg_or_zr( n, 0x1000 == ( imm5 & 0xffff ) ) );
            else if ( 0 == bits23_21 && !bit15 && 0 == bits14_11 && bit10 ) // DUP <Vd>.<T>, <Vn>.<Ts>[<index>]
            {
                uint64_t size = lowest_set_bit_nz( imm5 & 0xf );
                uint64_t index = get_bits( imm5, size + 1, 4 - ( size + 1 ) + 1 );
                uint64_t indsize = 64ull << get_bit( imm5, 4 );
                uint64_t esize = 8ull << size;
                uint64_t datasize = 64ull << Q;
                uint64_t elements = datasize / esize;
                tracer.Trace( "size %llu, index %llu, indsize %llu, esize %llu, datasize %llu, elements %llu\n", size, index, indsize, esize, datasize, elements );
                char byte_len = ( imm5 & 1 ) ? 'B' : ( 2 == ( imm5 & 3 ) ) ? 'H' : ( 4 == ( imm5 & 7 ) ) ? 'S' : ( 8 == ( imm5 & 0xf ) ) ? 'D' : '?';
                tracer.Trace( "dup v%llu.%s, v%llu.%c[%llu]\n", d, get_vector_T( imm5, Q ), n, byte_len, index );
            }
            else if ( bit21 && bit15 && 0 == bits14_11 && bit10 ) // ADD <Vd>.<T>, <Vn>.<T>, <Vm>.<T>.   add vector
            {
                uint64_t size = opbits( 22, 2 );
                uint64_t m = opbits( 16, 5 );
                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "add v%llu.%s, v%llu.%s, v%llu.%s\n", d, pT, n, pT, m, pT );
            }
            else if ( bit21 && 0xb == bits14_11 && 0 == bits20_16 && !bit15 ) // CNT
            {
                uint64_t size = opbits( 22, 2 );
                if ( 0 != size )
                    unhandled();

                const char * pT = get_ld1_vector_T( size, Q );
                tracer.Trace( "cnt v%llu.%s, v%llu.%s\n", d, pT, n, pT );
            }
            else if ( bit21 && 0x11 == bits20_16 && bit15 && 7 == bits14_11 ) // ADDV <V><d>, <Vn>.<T>
            {
                uint64_t size = opbits( 22, 2 );
                if ( 3 == size )
                    unhandled();
                const char * pT = get_ld1_vector_T( size, Q );
                char dstT = ( 0 == size ) ? 'B' : ( 1 == size ) ? 'H' : 'S';
                tracer.Trace( "addv %c%llu, v%llu.%s\n", dstT, d, n, pT );
            }
            else if ( bit21 && 1 == bits20_16 && !bit15 && 5 == bits14_11 && !bit10 ) // xtn, xtn2 XTN{2} <Vd>.<Tb>, <Vn>.<Ta>
            {
                uint64_t size = opbits( 22, 2 );
                if ( 3 == size )
                    unhandled();

                const char * pTb = get_ld1_vector_T( size, Q );
                const char * pTa = ( 0 == size ) ? "8h" : ( 1 == size ) ? "4s" : "2d";
                tracer.Trace( "xtn%s v%llu.%s, v%llu.%s\n", Q ? "2" : "", d, pTb, n, pTa );
            }
            else
            {
                tracer.Trace( "unknown opcode bits23_21 %lld, bit15 %llu, bits14_11 %llu, bit10 %llu\n", bits23_21, bit15, bits14_11, bit10 );
                unhandled();
            }
            break;
        }
        case 0x1e: // FMOV <Wd>, <Hn>    ;    FMUL                ;    FMOV <Wd>, imm       ;    FCVTZU <Wd>, <Dn>    ;    FRINTA <Dd>, <Dn>    ;    FMAXNM <Dd>, <Dn>, <Dm>, FMAX <Dd>, <Dn>, <Dm>
                   // FMAX <Dd>, <Dn>, <Dm> ; FMINNM <Dd>, <Dn>, <Dm>  ; FMIN <Dd>, <Dn>, <Dm> ; FRINTZ <Dd>, <Dn>    ;    FRINTP <Dd>, <Dn>
        case 0x9e: // FMOV <Xd>, <Hn>    ;    UCVTF <Hd>, <Dn>    ;    FCVTZU <Xd>, <Dn>    ;    FCVTAS <Xd>, <Dn>    ;    FCVTMU <Xd>, <Dn>
        {
            uint64_t sf = opbit( 31 );
            uint64_t ftype = opbits( 22, 2 );
            uint64_t bit21 = opbit( 21 );
            uint64_t bit11 = opbit( 11 );
            uint64_t bit10 = opbit( 10 );
            uint64_t bit4 = opbit( 4 );
            uint64_t bits21_19 = opbits( 19, 3 );
            uint64_t bits18_16 = opbits( 16, 3 );
            uint64_t bits18_10 = opbits( 10, 9 );
            uint64_t bits15_10 = opbits( 10, 6 );
            uint64_t n = opbits( 5, 5 );
            uint64_t d = opbits( 0, 5 );
            uint64_t rmode = opbits( 19, 2 );
            //tracer.Trace( "ftype %llu, bit21 %llu, rmode %llu, bits18_10 %#llx\n", ftype, bit21, rmode, bits18_10 );

            if ( 0x1e == hi8 && 4 == bits21_19 && 0x130 == bits18_10 ) // FRINTP <Dd>, <Dn>
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "frintp %c%llu, %c%llu\n", t, d, t, n );
            }
            else if ( 0x1e == hi8 && 4 == bits21_19 && 0x170 == bits18_10 ) // FRINTZ <Dd>, <Dn>
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "frintz %c%llu, %c%llu\n", t, d, t, n );
            }
            else if ( 0x1e == hi8 && bit21 && ( 0x12 == bits15_10 || 0x1a == bits15_10 ) ) // FMAX <Dd>, <Dn>, <Dm>    ;    FMAXNM <Dd>, <Dn>, <Dm>,
            {
                uint64_t m = opbits( 16, 5 );
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "%s %c%llu, %c%llu, %c%llu\n", ( 0x12 == bits15_10 ) ? "fmax" : "fmaxnm", t, d, t, n, t, m );
            }
            else if ( 0x1e == hi8 && bit21 && ( 0x16 == bits15_10 || 0x1e == bits15_10 ) ) // FMIN <Dd>, <Dn>, <Dm>    ;    FMINNM <Dd>, <Dn>, <Dm>
            {
                uint64_t m = opbits( 16, 5 );
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "%s %c%llu, %c%llu, %c%llu\n", ( 0x16 == bits15_10 ) ? "fmin" : "fminnm", t, d, t, n, t, m );
            }
            else if ( 0x1e == hi8 && 4 == bits21_19 && 0x150 == bits18_10 ) // FRINTM <Dd>, <Dn>
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "frintm %c%llu, %c%llu\n", t, d, t, n );
            }
            else if ( 0x1e == hi8 && bit21 && !bit11 && bit10 && bit4 ) // FCCMPE <Sn>, <Sm>, #<nzcv>, <cond>    ;    FCCMPE <Dn>, <Dm>, #<nzcv>, <cond>
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                uint64_t m = opbits( 16, 5 );
                uint64_t nzcv = opbits( 0, 4 );
                uint64_t cond = opbits( 12, 4 );
                tracer.Trace( "fccmpe %c%llu, %c%llu, #%#llx, %s\n", t, n, t, m, nzcv, get_cond( cond ) );
            }
            else if ( 3 == bits21_19 && 0 == bits18_16 ) // FCVTZS <Xd>, <Dn>, #<fbits>
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                uint64_t scale = opbits( 10, 6 );
                uint64_t fbits = 64 - scale;
                tracer.Trace( "fcvtzs %s, %c%llu, #%llu\n", reg_or_zr( d, sf ), t, n, fbits );
            }
            else if ( 6 == bits21_19 && 0x40 == bits18_10 ) // FCVTMU <Xd>, <Dn>
            {
                char type = ( 0 == ftype ) ? 's' : ( 1 == ftype ) ? 'd' : ( 3 == ftype ) ? 'h' : '?';
                tracer.Trace( "fcvtmu %s, %c%llu\n", reg_or_zr( d, sf ), type, n );
            }
            else if ( 4 == bits21_19 && 0x100 == bits18_10 ) // FCVTAS <Xd>, <Dn>
            {
                char type = ( 0 == ftype ) ? 's' : ( 1 == ftype ) ? 'd' : ( 3 == ftype ) ? 'h' : '?';
                tracer.Trace( "fcvtas %s, %c%llu\n", reg_or_zr( d, sf ), type, n );
            }
            else if ( 0x1e == hi8 && 4 == bits21_19 && 0x190 == bits18_10 ) // FRINTA <Dd>, <Dn>
            {
                char type = ( 0 == ftype ) ? 's' : ( 1 == ftype ) ? 'd' : ( 3 == ftype ) ? 'h' : '?';
                tracer.Trace( "frinta %c%llu, %c%llu\n", type, d, type, n );
            }
            else if ( ( 0x180 == ( bits18_10 & 0x1bf ) ) && ( bit21 ) && ( 0 == ( rmode & 2 ) ) ) // fmov reg, vreg  OR mov vreg, reg
            {
                uint64_t opcode = opbits( 16, 3 );
                if ( 0 == sf )
                {
                    if ( 0 != rmode )
                        unhandled();

                    if ( 3 == ftype )
                    {
                        if ( 6 == opcode )
                            tracer.Trace( "fmov w%llu, h%llu\n", d, n );
                        else if ( 7 == opcode )
                            tracer.Trace( "fmov h%llu, w%llu\n", d, n );
                    }
                    else if ( 0 == ftype )
                    {
                        if ( 7 == opcode )
                            tracer.Trace( "fmov s%llu, w%llu\n", d, n );
                        else if ( 6 == opcode )
                            tracer.Trace( "fmov w%llu, s%llu\n", d, n );
                    }
                    else
                        unhandled();
                }
                else
                {
                    if ( 0 == rmode )
                    {
                        if ( 3 == ftype && 6 == opcode )
                            tracer.Trace( "fmov x%llu, h%llu\n", d, n );
                        else if ( 3 == ftype && 7 == opcode )
                            tracer.Trace( "fmov h%llu, %s\n", d, reg_or_zr( n, false ) );
                        else if ( 1 == ftype && 7 == opcode )
                            tracer.Trace( "fmov d%llu, %s\n", d, reg_or_zr( n, true ) );
                        else if ( 1 == ftype && 6 == opcode )
                            tracer.Trace( "fmov x%llu, d%llu\n", d, n );
                        else
                            unhandled();
                    }
                    else
                    {
                        if ( 2 == ftype && 7 == opcode )
                            tracer.Trace( "fmov v%llu.D[1], x%llu\n", d, n );
                        else if ( 2 == ftype && 6 == opcode )
                            tracer.Trace( "fmov x%llu, v%llu.D[1]\n", d, n );
                    }
                }
            }
            else if ( 0x40 == bits18_10 && bit21 && 3 == rmode ) // FCVTZU <Wd>, <Dn>
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "fcvtzu %s, %c%llu\n", reg_or_zr( d, sf ), t, n );
            }
            else if ( 0x40 == ( bits18_10 & 0x1c0 ) && !bit21 && 3 == rmode ) // FCVTZU <Wd>, <Dn>, #<fbits>
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                uint64_t scale = opbits( 10, 6 );
                uint64_t fbits = 64 - scale;
                tracer.Trace( "fcvtzu %s, %cllu, #%llu\n", reg_or_zr( d, sf ), t, n, fbits );
            }
            else if ( ( 0x1e == hi8 ) && ( 4 == ( bits18_10 & 7 ) ) && bit21 && 0 == opbits( 5, 5 ) ) // fmov scalar immediate
            {
                //tracer.Trace( "ftype %llu, bit21 %llu, rmode %llu, bits18_10 %#llx\n", ftype, bit21, rmode, bits18_10 );
                uint64_t fltsize = ( 2 == ftype ) ? 64 : ( 8ull << ( ftype ^ 2 ) );
                char width = ( 3 == ftype ) ? 'H' : ( 0 == ftype ) ? 'S' : ( 1 == ftype ) ? 'D' : '?';
                uint64_t imm8 = opbits( 13, 8 );
                //tracer.Trace( "imm8: %llu == %#llx\n", imm8, imm8 );
                uint64_t val = vfp_expand_imm( imm8, fltsize );
                double dval = 0.0;
                if ( 1 == ftype )
                    mcpy( &dval, &val, sizeof( dval ) );
                else if ( 0 == ftype )
                {
                    float float_val;
                    mcpy( &float_val, &val, sizeof( float_val ) );
                    dval = (double) float_val;
                }
                tracer.Trace( "fmov %c%llu, #%lf // %#llx\n", width, d, dval, val );
            }
            else if ( ( 0x1e == hi8 ) && ( 2 == ( bits18_10 & 0x3f ) ) && ( bit21 ) ) // fmul vreg, vreg, vreg
            {
                uint64_t m = opbits( 16, 5 );
                if ( 0 == ftype ) // single-precision
                    tracer.Trace( "fmul s%llu, s%llu, s%llu\n", d, n, m );
                else if ( 1 == ftype ) // double-precision
                    tracer.Trace( "fmul d%llu, d%llu, d%llu\n", d, n, m );
                else
                    unhandled();
            }
            else if ( ( 0x1e == hi8 ) && ( 0x90 == ( bits18_10 & 0x19f ) ) && ( bit21 ) ) // fcvt vreg, vreg
            {
                uint64_t opc = opbits( 15, 2 );
                tracer.Trace( "fcvt %c%llu, %c%llu\n", get_fcvt_precision( opc ), d, get_fcvt_precision( ftype ), n );
            }
            else if ( ( 0x1e == hi8 ) && ( 0x10 == bits18_10 ) && ( 4 == bits21_19 ) ) // fmov vreg, vreg
            {
                tracer.Trace( "fmov %c%llu, %c%llu\n", get_fcvt_precision( ftype ), d, get_fcvt_precision( ftype ), n );
            }
            else if ( ( 0x1e == hi8 ) && ( 8 == ( bits18_10 & 0x3f ) ) && ( bit21 ) ) // fcmp vreg, vreg   OR    fcmp vreg, 0.0 and fcmpe variants
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t opc = opbits( 3, 2 );
                bool is_fcmpe = ( ( 3 == ftype && 2 == opc ) || ( 3 == ftype && 3 == opc ) || ( 0 == ftype && 2 == opc ) || ( 0 == ftype && 0 == m && 3 == opc ) ||
                                  ( 1 == ftype && 2 == opc ) || ( 1 == ftype && 0 == m && 3 == opc ) );
                if ( ( 1 == opc || 3 == opc ) && 0 == m )
                    tracer.Trace( "%s %c%llu, #0.0\n", is_fcmpe ? "fcmpe" : "fcmp", get_fcvt_precision( ftype ), n );
                else
                    tracer.Trace( "%s %c%llu, %c%llu\n", is_fcmpe ? "fcmpe" : "fcmp", get_fcvt_precision( ftype ), n, get_fcvt_precision( ftype ), m );
            }
            else if ( ( 0x1e == hi8 ) && ( 0x30 == bits18_10 ) && ( 4 == bits21_19 ) ) // fabs vreg, vreg
            {
                tracer.Trace( "fabs %c%llu, %c%llu\n", get_fcvt_precision( ftype ), d, get_fcvt_precision( ftype ), n );
            }
            else if ( 0x1e == hi8 && ( 6 == ( 0x3f & bits18_10 ) ) && bit21 ) // fdiv
            {
                uint64_t m = opbits( 16, 5 );
                if ( 0 == ftype ) // single-precision
                    tracer.Trace( "fdiv s%llu, s%llu, s%llu\n", d, n, m );
                else if ( 1 == ftype ) // double-precision
                    tracer.Trace( "fdiv d%llu, d%llu, d%llu\n", d, n, m );
                else
                    unhandled();
            }
            else if ( 0x1e == hi8 && ( 0xa == ( 0x3f & bits18_10 ) ) && bit21 ) // fadd
            {
                uint64_t m = opbits( 16, 5 );
                if ( 0 == ftype ) // single-precision
                    tracer.Trace( "fadd s%llu, s%llu, s%llu\n", d, n, m );
                else if ( 1 == ftype ) // double-precision
                    tracer.Trace( "fadd d%llu, d%llu, d%llu\n", d, n, m );
                else
                    unhandled();
            }
            else if ( 0x1e == hi8 && ( 0xe == ( 0x3f & bits18_10 ) ) && bit21 ) // fsub
            {
                uint64_t m = opbits( 16, 5 );
                if ( 0 == ftype ) // single-precision
                    tracer.Trace( "fsub s%llu, s%llu, s%llu\n", d, n, m );
                else if ( 1 == ftype ) // double-precision
                    tracer.Trace( "fsub d%llu, d%llu, d%llu\n", d, n, m );
                else
                    unhandled();
            }
            else if ( 0x80 == bits18_10 && bit21 && 0 == rmode ) // SCVTF (scalar, integer)
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "scvtf %c%llu, %s\n", t, d, reg_or_zr( n, sf ) );
            }
            else if ( 0x70 == bits18_10 && bit21 && 0 == rmode ) // fsqrt s#, s#
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "fsqrt %c%llu, %c%llu\n", t, d, t, n );
            }
            else if ( bit21 && ( 3 == ( 3 & bits18_10 ) ) ) // fcsel
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                uint64_t m = opbits( 16, 5 );
                uint64_t cond = opbits( 12, 4 );
                tracer.Trace( "fcsel %c%llu, %c%llu, %c%llu, %s\n", t, d, t, n, t, m, get_cond( cond ) );
            }
            else if ( bit21 && ( 0x50 == bits18_10 ) ) // fneg (scalar)
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "fneg %c%llu, %c%llu\n", t, d, t, n );
            }
            else if ( bit21 && 0 == bits18_10 && 3 == rmode ) // fcvtzs
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                tracer.Trace( "fcvtzs %s, %c%llu\n", reg_or_zr( d, sf ), t, n );
            }
            else if ( bit21 && ( 1 == ( bits18_10 & 3 ) ) && ( 0 == opbit( 4 ) ) ) // fccmp
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                uint64_t m = opbits( 16, 5 );
                uint64_t cond = opbits( 12, 4 );
                uint64_t nzcv = opbits( 0, 4 );
                tracer.Trace( "fccmp %c%llu, %c%llu, #%#llx, %s\n", t, n, t, m, nzcv, get_cond( cond ) );
            }
            else if ( bit21 && ( 0xc0 == ( 0x1c0 & bits18_10 ) ) && 0 == rmode ) // UCVTF <Hd>, <Wn>, #<fbits>
            {
                char t = ( 0 == ftype ) ? 's' : ( 3 == ftype ) ? 'h' : ( 1 == ftype ) ? 'd' : '?';
                uint64_t scale = opbits( 10, 6 );
                uint64_t fbits = 64 - scale;
                tracer.Trace( "ucvtf %c%llu, %s, #%#llx\n", t, d, reg_or_zr( n, sf ), fbits );
            }
            else
            {
                tracer.Trace( "ftype %llu, bit21 %llu, rmode %llu, bits18_10 %#llx\n", ftype, bit21, rmode, bits18_10 );
                unhandled();
            }
            break;
        }
        case 0x0c:
        case 0x4c: // LD1 { <Vt>.<T> }, [<Xn|SP>]    ;    LD2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
                   // ST2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]    ;    ST2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <imm>    ;    ST2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <Xm>
                   // LD3 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>]
                   // LD3 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <imm>
                   // LD3 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <Xm>
                   // LD4 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>]
                   // LD4 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <imm>
                   // LD4 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <Xm>
        {
            uint64_t Q = opbit( 30 );
            uint64_t L = opbit( 22 ); // load vs. store
            const char * pname = L ? "ld" : "st";
            uint64_t post_index = opbit( 23 );
            uint64_t opcode = opbits( 12, 4 );
            uint64_t size = opbits( 10, 2 );
            uint64_t bits23_21 = opbits( 21, 3 );
            uint64_t m = opbits( 16, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t t = opbits( 0, 5 );

            if ( 2 != bits23_21 && 6 != bits23_21 && 0 != bits23_21 && 4 != bits23_21 )
                unhandled();

            if ( ( 2 & opcode ) || 8 == opcode || 4 == opcode || 0 == opcode ) // LD1 / LD2 / LD3 / LD4 / ST1 / ST2 / ST3 / ST4
            {
                uint64_t t2 = ( t + 1 ) % 32;
                if ( post_index )
                {
                    if ( 31 == m )
                    {
                        const char * pT = get_ld1_vector_T( size, Q );
                        if ( 7 == opcode ) // LD1 { <Vt>.<T> }, [<Xn|SP>], <imm>
                            tracer.Trace( "%s1 {v%llu.%s}, [%s], #%llu\n", pname, t, pT, reg_or_sp( n, true ), Q ? 16ull : 8ull );
                        else if ( 8 == opcode ) // LD2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <imm>
                            tracer.Trace( "%s2 {v%llu.%s, v%llu.%s}, [%s], #%llu\n", pname, t, pT, t2, pT, reg_or_sp( n, true ), Q ? 32llu : 16llu );
                        else if ( 3 == opcode ) // LD3 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <imm>
                            tracer.Trace( "%s3 {v%llu.%s-v%llu.%s}, [%s], #%llu\n", pname, t, pT, ( t + 2 ) % 32, pT, reg_or_sp( n, true ), Q ? 64llu : 32llu );
                        else if ( 0 == opcode ) // LD4 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <imm>
                            tracer.Trace( "%s4 {v%llu.%s-v%llu.%s}, [%s], #%llu\n", pname, t, pT, ( t + 3 ) % 32, pT, reg_or_sp( n, true ), Q ? 64llu : 32llu );
                        else
                            unhandled();
                    }
                    else
                        unhandled();
                }
                else // no offset
                {
                    if ( 0 == m )
                    {
                        const char * pT = get_ld1_vector_T( size, Q );
                        if ( 7 == opcode ) // LD1 { <Vt>.<T> }, [<Xn|SP>]
                            tracer.Trace( "%s1 {v%llu.%s}, [%s]\n", pname, t, pT, reg_or_sp( n, true ) );
                        else if ( 10 == opcode ) // LD1 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
                            tracer.Trace( "%s1 {v%llu.%s}, {v%llu.%s}, [%s]\n", pname, t, pT, t2, pT, reg_or_sp( n, true ) );
                        else if ( 8 == opcode ) // LD2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
                            tracer.Trace( "%s2 { v%llu.%s, %llu.%s }, [%s]\n", pname, t, pT, t2, pT, reg_or_sp( n, true ) );
                        else if ( 4 == opcode ) // LD3 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
                            tracer.Trace( "%s3 { v%llu.%s-v%llu.%s }, [%s]\n", pname, t, pT, ( t + 2 ) % 32, pT, reg_or_sp( n, true ) );
                        else if ( 0 == opcode || 2 == opcode ) // LD4 { <Vt>.<T>-<Vtn>.<T> }, [<Xn|SP>]
                            tracer.Trace( "%s4 { v%llu.%s-v%llu.%s }, [%s]\n", pname, t, pT, ( t + 3 ) % 32, pT, reg_or_sp( n, true ) );
                        else
                            unhandled();
                    }
                    else
                        unhandled();
                }
            }
            else if ( 0 == opcode && 0 == opbits( 12, 9 ) ) // LD4 multiple structures
            {
                if ( 2 == bits23_21 ) // no offset LD4 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>]
                {
                    const char * pT = get_ld1_vector_T( size, Q );
                    tracer.Trace( "ld4 {v%llu.%s-v%llu.%s}, [%s]\n", t, pT, ( t + 3 ) % 32, pT, reg_or_sp( n , true ) );
                }
                else if ( 6 == bits23_21 ) // post-index
                    unhandled();
                else
                    unhandled();
            }
            else
                unhandled();
            break;
        }
        case 0x88: // LDAXR <Wt>, [<Xn|SP>{, #0}]    ;    LDXR <Wt>, [<Xn|SP>{, #0}]    ;    STXR <Ws>, <Wt>, [<Xn|SP>{, #0}]    ;    STLXR <Ws>, <Wt>, [<Xn|SP>{, #0}]
                   //                                                                        STLR <Wt>, [<Xn|SP>{, #0}]          ;    STLR <Wt>, [<Xn|SP>, #-4]!
        case 0xc8: // LDAXR <Xt>, [<Xn|SP>{, #0}]    ;    LDXR <Xt>, [<Xn|SP>{, #0}]    ;    STXR <Ws>, <Xt>, [<Xn|SP>{, #0}]    ;    STLXR <Ws>, <Xt>, [<Xn|SP>{, #0}]
                   //                                                                        STLR <Xt>, [<Xn|SP>{, #0}]          ;    STLR <Xt>, [<Xn|SP>, #-8]!
        {
            uint64_t t = opbits( 0, 5 );
            uint64_t n = opbits( 5, 5 );
            uint64_t t2 = opbits( 10, 5 );
            uint64_t s = opbits( 16, 5 );
            uint64_t L = opbits( 21, 2 );
            uint64_t oO = opbit( 15 );
            uint64_t bit23 = opbit( 23 );
            uint64_t bit30 = opbit( 30 );

            if ( 0x1f != t2 )
                unhandled();

            if ( 0 == L ) // stxr, stlr
            {
                if ( bit23 )
                    tracer.Trace( "stlr %s, [%s]\n", reg_or_zr( t, bit30 ), reg_or_sp( n, bit30 ) );
                else
                    tracer.Trace( "%s %s, %s, [ %s ]\n", ( 1 == oO ) ? "stlxr" : "stxr", reg_or_zr( s, false ), reg_or_zr( t, ( 0xc8 == hi8 ) ), reg_or_sp( n, true ) );
            }
            else if ( 2 == L ) // ldxr and ldaxr
            {
                if ( 0x1f != s )
                    unhandled();
                tracer.Trace( "%s %s, [ %s ]\n", ( 1 == oO ) ? "ldaxr" : "ldxr", reg_or_zr( t, ( 0xc8 == hi8 ) ), reg_or_sp( n, true ) );
            }
            break;
        }
        case 0xd6: // BLR <Xn>    ;    BR <Xn>    ;    RET {<Xn>}
        {
            uint64_t n = opbits( 5, 5 );
            uint64_t theop = opbits( 21, 2 );
            uint64_t bit23 = opbit( 23 );
            uint64_t op2 = opbits( 12, 9 );
            uint64_t A = opbit( 11 );
            uint64_t M = opbit( 10 );
            if ( 0 != bit23 )
                unhandled();
            if ( 0x1f0 != op2 )
                unhandled();
            if ( ( 0 != A ) || ( 0 != M ) )
                unhandled();

            if ( 0 == theop ) // br
                tracer.Trace( "br x%llu\n", n );
            else if ( 1 == theop ) // blr
                tracer.Trace( "blr x%llu\n", n );
            else if ( 2 == theop ) // ret
                tracer.Trace( "ret x%llu\n", n );
            else
                unhandled();
            break;
        }
        case 0x72: // MOVK <Wd>, #<imm>{, LSL #<shift>}       ;  ANDS <Wd>, <Wn>, #<imm>
        case 0xf2: // MOVK <Xd>, #<imm>{, LSL #<shift>}       ;  ANDS <Xd>, <Xn>, #<imm>
        {
            uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
            uint64_t bit23 = opbit( 23 ); // 1 for MOVK, 0 for ANDS
            if ( bit23 ) // MOVK
            {
                uint64_t hw = opbits( 21, 2 );
                uint64_t pos = ( hw << 4 );
                uint64_t imm16 = opbits( 5, 16 );
                uint64_t d = opbits( 0, 5 );
                tracer.Trace( "movk %s, #%#llx, LSL #%llu\n", reg_or_zr( d, xregs ), imm16, pos );
            }
            else // ANDS
            {
                uint64_t N_immr_imms = opbits( 10, 13 );
                uint64_t op2 = decode_logical_immediate( N_immr_imms, xregs ? 64 : 32 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                tracer.Trace( "ands %s, %s, #%#llx\n", reg_or_zr( d, xregs ), reg_or_zr( n, xregs ), op2 );
            }
            break;
        }
        case 0x38: // B
        case 0x78: // H
        case 0xb8: // W
        case 0xf8: // X
        {
            // LDR <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
            // LDR <Xt>, [<Xn|SP>], #<simm>
            // LDR <Xt>, [<Xn|SP>, #<simm>]!
            // STR <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
            // STR <Xt>, [<Xn|SP>], #<simm>
            // STR <Xt>, [<Xn|SP>, #<simm>]!
            // W, H and B variants use <Wt> as the first argument
            // H and B variants use LDRH, STRH, LDRB, STRB instructions. W and X variants use STR and LDR
            // LDR has sign-extend LDRSx and LDURSW variants

            uint64_t opc = opbits( 21, 3 );
            uint64_t n = opbits( 5, 5 );
            uint64_t t = opbits( 0, 5 );
            bool xregs = ( 0 != opbit( 30 ) );

            const char * suffix = "";
            if ( 0x38 == hi8 )
                suffix = "b";
            else if ( 0x78 == hi8 )
                suffix = "h";

            char prefix = 'w';
            if ( xregs )
                prefix = 'x';

            if ( 0 == opc ) // str (immediate) post-index and pre-index
            {
                uint64_t unsigned_imm9 = opbits( 12, 9 );
                int64_t extended_imm9 = sign_extend( unsigned_imm9, 8 );
                uint64_t option = opbits( 10, 2 );
                if ( 0 == option) // // STUR <Xt>, [<Xn|SP>{, #<simm>}]
                    tracer.Trace( "stur%s %s, %s, #%lld // so\n", suffix, reg_or_zr( t, xregs ), reg_or_sp( n, xregs ), extended_imm9 );
                else if ( 1 == option) // post-index STR <Xt>, [<Xn|SP>], #<simm>
                    tracer.Trace( "str%s %c%llu, %s, #%lld // po\n", suffix, prefix, t, reg_or_sp( n, true ), extended_imm9 );
                else if ( 3 == option ) // pre-index STR <Xt>, [<Xn|SP>, #<simm>]!
                    tracer.Trace( "str%s %c%llu, [%s, #%lld]! //pr\n", suffix, prefix, t, reg_or_sp( n, true ), extended_imm9 );
                else
                    unhandled();
            }
            else if ( 1 == opc ) // STR <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t shift = opbit( 12 );
                uint64_t option = opbits( 13, 3 );
                tracer.Trace( "str%s %s, [ %s, x%llu, %s #%llu]\n", suffix, reg_or_zr( t, xregs ), reg_or_sp( n, true ), m, extend_type( option ),
                              ( 3 == option ) ? 0 : ( 0 == shift ) ? 0ull : xregs ? 3ull : 2ull );
            }
            else if ( 2 == opc ) // ldr (immediate)
            {
                uint64_t unsigned_imm9 = opbits( 12, 9 );
                int64_t extended_imm9 = sign_extend( unsigned_imm9, 8 );
                uint64_t option = opbits( 10, 2 );
                if ( 0 == option) // LDUR <Xt>, [<Xn|SP>{, #<simm>}]
                    tracer.Trace( "ldur%s %c%llu, [%s, #%lld] //so\n", suffix, prefix, t, reg_or_sp( n, true ), extended_imm9 );
                else if ( 1 == option) // post-index LDR <Xt>, [<Xn|SP>], #<simm>
                    tracer.Trace( "ldr%s %c%llu, [%s], #%lld //po\n", suffix, prefix, t, reg_or_sp( n, true ), extended_imm9 );
                else if ( 3 == option ) // pre-index LDR <Xt>, [<Xn|SP>, #<simm>]!
                    tracer.Trace( "ldr%s %c%llu, [%s, #%lld]! //pr\n", suffix, prefix, t, reg_or_sp( n, true ), extended_imm9 );
                else
                    unhandled();
            }
            else if ( 3 == opc ) // LDR <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t shift = opbit( 12 );
                uint64_t option = opbits( 13, 3 );
                tracer.Trace( "ldr%s %s, [%s, %s, %s #%llu]\n", suffix, reg_or_zr( t, xregs ), reg_or_sp( n, true ), reg_or_zr( m, true ),
                              extend_type( option ), ( 0 == shift ) ? 0ull : xregs ? 3ull : 2ull );
            }
            else if ( 4 == opc || 6 == opc ) // LDRSW <Xt>, [<Xn|SP>], #<simm>    ;    LDRSW <Xt>, [<Xn|SP>, #<simm>]!
            {
                uint64_t bits11_10 = opbits( 10, 2 );
                if ( 0 == bits11_10 ) // LDURSB <Wt>, [<Xn|SP>{, #<simm>}]    ;    LDURSB <Xt>, [<Xn|SP>{, #<simm>}]
                {
                    bool isx = ( 0 == opbit( 22 ) );
                    int64_t imm9 = sign_extend( opbits( 12, 9 ), 8 );
                    tracer.Trace( "ldurs%s %s, [%s, #%lld]\n", suffix, reg_or_zr( t, isx ), reg_or_sp( n, true ), imm9 );
                }
                else
                {
                    uint64_t preindex = opbit( 11 ); // 1 for pre, 0 for post increment
                    int64_t imm9 = sign_extend( opbits( 12, 9 ), 8 );
                    xregs = ( 4 == opc );
                    if ( preindex )
                        tracer.Trace( "ldrs%s %s [%s, #%lld]! // pr\n", suffix, reg_or_zr( t, xregs ), reg_or_sp( n, true ), imm9 );
                    else
                        tracer.Trace( "ldrs%s %s [%s], #%lld // po\n", suffix, reg_or_zr( t, xregs ), reg_or_sp( n, true ), imm9 );
                }
            }
            else if ( 5 == opc  || 7 == opc ) // hi8 = 0x78
                                              //     (opc == 7)                  LDRSH <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
                                              //     (opc == 5)                  LDRSH <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
                                              // hi8 = 0x38
                                              //     (opc == 7 && option != 011) LDRSB <Wt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
                                              //     (opc == 5 && option != 011) LDRSB <Xt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
                                              //     (opc == 7 && option == 011) LDRSB <Wt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
                                              //     (opc == 5 && option == 011) LDRSB <Xt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
                                              // hi8 == 0xb8
                                              //     (opc == 5 && option = many) LDRSW <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
            {
                uint64_t m = opbits( 16, 5 );
                uint64_t shift = opbit( 12 );
                uint64_t option = opbits( 13, 3 );
                bool mIsX = ( 1 == ( option & 1 ) );
                bool tIsX = ( 5 == opc );

                if ( 0xb8 == hi8 )
                    tracer.Trace( "ldrsw %s, [%s, %s, %s, %llu]\n", reg_or_zr( t, true ), reg_or_sp( n, true ), reg_or_zr( m, option & 1 ), extend_type( option ), shift ? 2 : 0 );
                else if ( 0x38 == hi8 )
                {
                    if ( 3 == option )
                        tracer.Trace( "ldrsb %s, [%s, x%llu {, LSL %lluu}]\n", reg_or_zr( t, tIsX ), reg_or_sp( n, true ), m, shift );
                    else
                        tracer.Trace( "ldrsb %s, [%s, %s, %s {#%llu}]\n", reg_or_zr( t, tIsX ), reg_or_sp( n, true ), reg_or_zr( m, mIsX ), extend_type( option ), shift );
                }
                else if ( 0x78 == hi8 )
                    tracer.Trace( "ldrsh %s, [%s, %s {, %s #%llu}]\n", reg_or_zr( t, tIsX ), reg_or_sp( n, true ),
                                  reg_or_zr( m, mIsX ), extend_type( option ), shift );
                else
                    unhandled();
            }
            else
                unhandled();

            break;
        }
        case 0x39: // B
        case 0x79: // H                              ;    LDRSH <Wt>, [<Xn|SP>{, #<pimm>}]
        case 0xb9: // W
        case 0xf9: // X ldr + str unsigned offset    ;    LDRSW <Xt>, [<Xn|SP>{, #<pimm>}]
        {
            // LDR <Xt>, [<Xn|SP>{, #<pimm>}]
            // STR <Xt>, [<Xn|SP>{, #<pimm>}]

            uint64_t opc = opbits( 22, 2 );
            uint64_t imm12 = opbits( 10, 12 );
            uint64_t lsl = opbits( 30, 2 );
            imm12 <<= lsl;
            uint64_t t = opbits( 0, 5 );
            uint64_t n = opbits( 5, 5 );

            const char * suffix = "";
            if ( 0x39 == hi8 )
                suffix = "b";
            else if ( 0x79 == hi8 )
                suffix = "h";

            char prefix = 'w';
            if ( 0xf9 == hi8 )
                prefix = 'x';

            if ( 0 == opc )
                tracer.Trace( "str%s %s, [%s,#%llu] //uo\n", suffix, reg_or_zr( t, ( 0xf9 == hi8 ) ), reg_or_sp( n, true ), imm12 );
            else if ( 1 == opc )
                tracer.Trace( "ldr%s %c%llu, [%s,#%llu] //uo\n", suffix, prefix, t, reg_or_sp( n, true ), imm12 );
            else if ( 2 == opc || 3 == opc )
                tracer.Trace( "ldrs%s %c%llu, [%s,#%llu] //uo\n", suffix, prefix, t, reg_or_sp( n, true ), imm12 );
            else
                unhandled();

            break;
        }
        default:
            unhandled();
    }

    static char acregs[ 32 * 32 + 10 ]; // way too much.
    acregs[ 0 ] = 0;
    int len = 0;
    for ( int r = 0; r < 31; r++ )
        if ( 0 != regs[ r ] )
            len += snprintf( & acregs[ len ], 32, "%u:%llx ", r, regs[ r ] );
    len += snprintf( &acregs[ len ], 32, "sp:%llx", regs[ 31 ] );
    tracer.Trace( "         %s\n", acregs );
} //trace_state

// N (negative): Set if the result is negative
// Z (zero): Set if the result is zero
// C (carry): Set if the result cannot be represented as an unsigned integer
// V (overflow): Set if the result cannot be represented as a signed integer

uint64_t Arm64::add_with_carry64( uint64_t x, uint64_t y, bool carry, bool setflags )
{
    uint64_t result = x + y + (uint64_t) carry;

    if ( setflags )
    {
        int64_t iresult = (int64_t) result;
        fN = ( iresult < 0 );
        fZ = ( 0 == result );

        // strangely literal fC computation; there must be a faster way

        uint64_t u_y = y + (uint64_t) carry;
        uint64_t u_low = ( ( x & 0xffffffff ) + ( u_y & 0xffffffff ) );
        uint64_t u_low_carry = ( u_low >> 32 );
        uint64_t carry_carry = ( ~0ull == y && carry ) ? 1 : 0;
        uint64_t u_hi = ( ( x >> 32 ) + ( u_y >> 32 ) + u_low_carry + carry_carry );
        uint64_t u_sum = ( u_hi << 32 ) | ( 0xffffffff & u_low );
        fC = ( ( result != u_sum ) || ( 0 != ( u_hi >> 32 ) ) );

        int64_t ix = (int64_t) x;
        int64_t iy = (int64_t) y;
        fV = ( ( ( ix >= 0 && iy >= 0 ) && ( iresult < ix || iresult < iy ) ) ||
               ( ( ix < 0 && iy < 0 ) && ( iresult > ix || iresult > iy ) ) );
    }
    return result;
} //add_with_carry64

__inline_perf uint64_t Arm64::sub64( uint64_t x, uint64_t y, bool setflags )
{
    return add_with_carry64( x, ~y, true, setflags );
} //sub64

uint32_t Arm64::add_with_carry32( uint32_t x, uint32_t y, bool carry, bool setflags )
{
    uint64_t unsigned_sum = (uint64_t) x + (uint64_t) y + (uint64_t) carry;
    uint32_t result = (uint32_t) unsigned_sum;

    if ( setflags )
    {
        // this method of setting flags is as the Arm documentation suggests
        fN = ( ( (int32_t) result ) < 0 );
        fZ = ( 0 == result );
        fC = ( (uint64_t) result != unsigned_sum );
        int64_t signed_sum = (int64_t) (int32_t) x + (int64_t) (int32_t) y + (int64_t) carry;
        fV = ( ( (int64_t) (int32_t) result ) != signed_sum );
    }
    return result;
} //add_with_carry32

__inline_perf uint32_t Arm64::sub32( uint32_t x, uint32_t y, bool setflags )
{
    return add_with_carry32( x, ~y, true, setflags );
} //sub32

uint64_t Arm64::shift_reg64( uint64_t reg, uint64_t shift_type, uint64_t amount )
{
    uint64_t val = val_reg_or_zr( reg );
    amount &= 0x3f;
    if ( 0 == amount )
        return val;

    if ( 0 == shift_type ) // lsl
        val <<= amount;
    else if ( 1 == shift_type ) // lsr
        val >>= amount;
    else if ( 2 == shift_type ) // asr
        val = (uint64_t) ( ( (int64_t) val ) >> amount ); // modern C compilers do the right thing
    else if ( 3 == shift_type ) // ror.
        val = ( ( val >> amount ) | ( val << ( 64 - amount ) ) );
    else
        unhandled();

    return val;
} //shift_reg64

uint32_t Arm64::shift_reg32( uint64_t reg, uint64_t shift_type, uint64_t amount )
{
    uint32_t val = ( 31 == reg ) ? 0 : (uint32_t) regs[ reg ];
    amount &= 0x1f;
    if ( 0 == amount )
        return val;

    if ( 0 == shift_type ) // lsl
        val <<= amount;
    else if ( 1 == shift_type ) // lsr
        val >>= amount;
    else if ( 2 == shift_type ) // asr
        val = (uint32_t) ( (int32_t) val >> amount ); // modern C compilers do the right thing
    else if ( 3 == shift_type ) // ror.
        val = ( ( val >> amount ) | ( val << ( 32 - amount ) ) );
    else
        unhandled();

    return val;
} //shift_reg32

double set_double_sign( double d, bool sign )
{
    uint64_t val = sign ? ( ( * (uint64_t *) &d ) | 0x8000000000000000 ) : ( ( * (uint64_t *) &d ) & 0x7fffffffffffffff );
    return * (double *) &val;
} //set_double_sign

double do_fsub( double a, double b )
{
    if ( isinf( a ) && isinf( b ) )
    {
        if ( signbit( a ) != signbit( b ) )
            return a;
        return MY_NAN; // msft C will return -nan if this check isn't here
    }

    if ( isnan( a ) )
        return a;

    if ( isnan( b ) )
        return b;

    return a - b;
} //do_fsub

double do_fadd( double a, double b )
{
    bool ainf = isinf( a );
    bool binf = isinf( b );

    if ( ainf && binf )
    {
        if ( signbit( a ) == signbit( b ) )
            return a;
        return MY_NAN; // msft C will return -nan if this check isn't here
    }

    if ( isnan( a ) )
        return a;

    if ( isnan( b ) )
        return b;

    if ( ainf )
        return a;

    if ( binf )
        return b;

    return a + b;
} //do_fadd

double do_fmul( double a, double b )
{
    if ( isnan( a ) )
        return a;

    if ( isnan( b ) )
        return b;

    bool ainf = isinf( a );
    bool binf = isinf( b );
    bool azero = ( 0.0 == a );
    bool bzero = ( 0.0 == b );

    if ( ( ainf && bzero ) || ( azero && binf ) )
        return MY_NAN;

    if ( ainf || binf )
        return set_double_sign( INFINITY, signbit( a ) != signbit( b ) );

    if ( azero || bzero )
        return set_double_sign( 0.0, signbit( a ) != signbit( b ) );

    return a * b;
} //do_fmul

double do_fdiv( double a, double b )
{
    if ( isnan( a ) )
        return a;

    if ( isnan( b ) )
        return b;

    bool ainf = isinf( a );
    bool binf = isinf( b );
    bool azero = ( 0.0 == a );
    bool bzero = ( 0.0 == b );

    if ( ( ainf && binf ) || ( azero && bzero ) )
        return MY_NAN;

    if ( ainf )
        return set_double_sign( INFINITY, signbit( a ) != signbit( b ) );

    if ( binf || azero )
        return set_double_sign( 0.0, signbit( a ) != signbit( b ) );

    return a / b;
} //do_fdiv

double do_fmin( double a, double b )
{
    if ( ( 0.0 == a ) && ( 0.0 == b ) )
    {
        if ( signbit( a ) )
            return a;
        return b;
    }

    bool anan = isnan( a );
    bool bnan = isnan( b );

    if ( anan && bnan )
        return a;         // not clearly documented, but this is what hardware does

    if ( anan )
        return b;

    if ( bnan )
        return a;

    return get_min( a, b );
} //do_fmin

double do_fmax( double a, double b )
{
    if ( ( 0.0 == a ) && ( 0.0 == b ) )
    {
        if ( signbit( a ) )
            return b;
        return a;
    }

    bool anan = isnan( a );
    bool bnan = isnan( b );

    if ( anan && bnan )
        return a;         // not clearly documented, but this is what hardware does

    if ( anan )
        return b;

    if ( bnan )
        return a;

    return get_max( a, b );
} //do_fmax

bool Arm64::check_conditional( uint64_t cond ) const
{
    assert( cond <= 15 );

    switch ( cond & 0xf ) // do the reduant mask so the msft compiler doesn't add a conditional
    {
        case 0: { return fZ; }                          // EQ = Zero / Equal
        case 1: { return !fZ; }                         // NE = Not Equal
        case 2: { return fC; }                          // CS = Carry Set
        case 3: { return !fC; }                         // CC = Carry Clear
        case 4: { return fN; }                          // MI = Minus / Negative
        case 5: { return !fN; }                         // PL = Plus. Positive or Zero
        case 6: { return fV;  }                         // VS = Overflow Set
        case 7: { return !fV;  }                        // VC = Overflow Clear
        case 8: { return ( fC && !fZ ); }               // HI = Unsigned Higher
        case 9: { return ( !fC || fZ ); }               // LS = Lower or Same
        case 10: { return ( fN == fV ); }               // GE = Signed Greater Than or Equal
        case 11: { return ( fN != fV ); }               // LT = Signed Less Than
        case 12: { return ( ( fN == fV ) && !fZ ); }    // GT = Signed Greater Than
        case 13: { return ( ( fN != fV ) || fZ ); }     // LE = Signed Less Than or Equal
        case 14: case 15:
        default: { return true; }                       // AL = Always true. not used in practice. 14 and 15
    }
} //check_conditional

void Arm64::set_flags_from_double( double result )
{
    if ( isnan( result ) )
    {
        fN = fZ = false;
        fC = fV = true;
    }
    else if ( result > 0.0 )
    {
        fN = fZ = fV = false;
        fC = true;
    }
    else if ( result < 0.0 )
    {
        fN = true;
        fZ = fC = fV = false;
    }
    else
    {
        fN = fV = false;
        fZ = fC = true;
    }
} //set_flags_from_double

#ifdef _WIN32
__declspec(noinline)
#endif
void Arm64::force_trace_vregs()
{
    for ( uint64_t i = 0; i < _countof( vregs ); i++ )
    {
        if ( memcmp( &vec_zeroes, &vregs[ i ], sizeof( vec_zeroes ) ) )
        {
            tracer.Trace( "    vreg %2llu: ", i );
            tracer.TraceBinaryData( (uint8_t *) & vregs[ i ], 16, 4 );
        }
    }
} //trace_vregs

void Arm64::trace_vregs()
{
    if ( !tracer.IsEnabled() ) // can happen when an app enables instruction tracing via a syscall but overall tracing is turned off.
        return;

    if ( ! ( g_State & stateTraceInstructions ) )
        return;

    force_trace_vregs();
} //trace_vregs

uint64_t Arm64::run( void )
{
    cycles = 0;

    for ( ;; )
    {
        #ifndef NDEBUG
            if ( regs[ 31 ] <= ( stack_top - stack_size ) )
                emulator_hard_termination( *this, "stack pointer is below stack memory:", regs[ 31 ] );

            if ( regs[ 31 ] > stack_top )
                emulator_hard_termination( *this, "stack pointer is above the top of its starting point:", regs[ 31 ] );

            if ( pc < base )
                emulator_hard_termination( *this, "pc is lower than memory:", pc );

            if ( pc >= ( base + mem_size - stack_size ) )
                emulator_hard_termination( *this, "pc is higher than it should be:", pc );

            if ( 0 != ( regs[ 31 ] & 0xf ) ) // by convention, arm64 stacks are 16-byte aligned
                emulator_hard_termination( *this, "the stack pointer isn't 16-byte aligned:", regs[ 31 ] );
        #endif

        op = getui32( pc );
        uint8_t hi8 = (uint8_t) ( op >> 24 );

        if ( 0 != g_State )
        {
            if ( g_State & stateEndEmulation )
            {
                g_State &= ~stateEndEmulation;
                break;
            }

            if ( g_State & stateTraceInstructions )
                trace_state();
        }

        cycles++;

        switch ( hi8 )
        {
            case 0: // UDF
            {
                uint64_t bits23to16 = opbits( 16, 8 );
                if ( 0 == bits23to16 )
                {
                    uint64_t imm16 = op & 0xffff;
                    emulator_hard_termination( *this, "permanently undefined instruction encountered", imm16 );
                }
                else
                    unhandled();
                break;
            }
            case 0x5f: // SHL D<d>, D<n>, #<shift>    ;    FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]    ;    FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                       // SSHR D<d>, D<n>, #<shift>
            {
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t opcode = opbits( 10, 6 );
                uint64_t immh = opbits( 19, 4 );
                uint64_t immb = opbits( 16, 3 );
                uint64_t bit23 = opbit( 23 );
                uint64_t bit22 = opbit( 22 );
                uint64_t bits15_10 = opbits( 10, 6 );

                if ( !bit23 && bit22 && 0x15 == opcode ) // SHL D<d>, D<n>, #<shift>
                {
                    uint64_t esize = 8ull << 3;
                    uint64_t shift = ( ( immh << 3 ) | immb ) - esize;
                    vregs[ d ].set64( 0, ( vregs[ n ].get64( 0 ) << shift ) );
                    vregs[ d ].set64( 1, 0 );
                }
                else if ( bit23 && ( 4 == opcode || 6 == opcode || 0x24 == opcode || 0x26 == opcode ) ) // FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]    ;    FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                {
                    uint64_t m = opbits( 16, 5 );
                    uint64_t sz = opbit( 22 );
                    uint64_t L = opbit( 21 );
                    uint64_t M = opbit( 20 );
                    uint64_t H = opbit( 11 );
                    uint64_t Q = opbit( 20 );
                    uint64_t Rmhi = M;
                    if ( 0 == sz )
                        Rmhi = ( H << 1 ) | L;
                    else if ( 0 == L )
                        Rmhi = H;
                    else
                        unhandled();
                    uint64_t esize = 32ull << sz;
                    uint64_t ebytes = esize / 8;
                    uint64_t datasize = 64ull << Q;
                    uint64_t elements = datasize / esize;
                    vec16_t target;
                    if ( 4 == opcode || 6 == opcode ) // FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                        target = vregs[ d ];
                    vec16_t & nvec = vregs[ n ];

                    float mfloat = 0.0f;
                    double mdouble = 0.0;
                    if ( 4 == ebytes )
                        mfloat = vregs[ m ].getf( Rmhi );
                    else if ( 8 == ebytes )
                        mdouble = vregs[ m ].getd( Rmhi );

                    for ( uint64_t e = 0; e < elements; e++ )
                    {
                        if ( 4 == ebytes )
                            target.setf( e, target.getf( e ) + (float) do_fmul( nvec.getf( e ), mfloat ) );
                        else if ( 8 == ebytes )
                            target.setd( e, target.getd( e ) + do_fmul( nvec.getd( e ), mdouble ) );
                    }

                    vregs[ d ] = target;
                }
                else if ( !bit23 && bit22 && 1 == bits15_10 ) // SSHR D<d>, D<n>, #<shift>
                {
                    uint64_t shift = ( 32ull * 2 ) - ( ( immh << 3 ) | immb );
                    vregs[ d ].set32( 0, ( (int32_t) vregs[ n ].get32( 0 ) ) >> shift );
                }
                else
                    unhandled();
                break;
            }
            case 0x0d: case 0x4d: // LD1 { <Vt>.B }[<index>], [<Xn|SP>]    ;    LD1 { <Vt>.B }[<index>], [<Xn|SP>], #1
                                  // LD1R { <Vt>.<T> }, [<Xn|SP>], <imm>   ;    LD1R { <Vt>.<T> }, [<Xn|SP>], <Xm>
                                  // ST1 { <Vt>.B }[<index>], [<Xn|SP>]    ;    ST1 { <Vt>.B }[<index>], [<Xn|SP>], #1
            {
                uint64_t R = opbit( 21 );
                if ( R )
                    unhandled();
                uint64_t post_index = opbit( 23 );
                uint64_t opcode = opbits( 13, 3 );
                uint64_t bit13 = opbit( 13 );
                if ( bit13 )
                    unhandled();
                uint64_t size = opbits( 10, 2 );
                uint64_t n = opbits( 5, 5 );
                uint64_t m = opbits( 16, 5 );
                uint64_t t = opbits( 0, 5 );
                uint64_t S = opbit( 12 );
                uint64_t Q = opbit( 30 );
                uint64_t L = opbit( 22 ); // load vs. store
                uint64_t index = 0;
                uint64_t replicate = ( 6 == opcode );
                uint64_t scale = get_bits( opcode, 1, 2 );
                if ( 3 == scale )
                    scale = size;
                else if ( 0 == scale )
                    index = ( Q << 3 ) | ( S << 2 ) | size;
                else if ( 1 == scale )
                    index = ( Q << 2 ) | ( S << 1 ) | get_bit( size, 1 );
                else if ( 2 == scale )
                {
                    if ( 0 == ( size & 1 ) )
                        index = ( Q << 1 ) | S;
                    else
                    {
                        index = Q;
                        scale = 3;
                    }
                }

                uint64_t esize = 8ull << scale;
                uint64_t ebytes = esize / 8;
                uint64_t offs = 0;
                uint64_t selem = ( ( opcode & 1 ) << 1 ) + 1;
                uint64_t nval = regs[ n ];

                if ( replicate )
                {
                    if ( !L )
                        unhandled();

                    for ( uint64_t e = 0; e < selem; e++ )
                    {
                        uint64_t eaddr = nval + offs;
                        uint64_t element = 0;
                        if ( 1 == ebytes )
                            element = * getmem( eaddr );
                        else if ( 2 == ebytes )
                            element = getui16( eaddr );
                        else if ( 4 == ebytes )
                            element = getui32( eaddr );
                        else
                            element = getui64( eaddr );
                        element = replicate_bytes( element, ebytes );
                        vregs[ t ].set64( 0, element );
                        vregs[ t ].set64( 1, Q ? element : 0 );
                        offs += ebytes;
                        t = ( ( t + 1 ) % 32 );
                    }
                }
                else
                {
                    for ( uint64_t e = 0; e < selem; e++ )
                    {
                        uint64_t eaddr = nval + offs;
                        if ( L )
                            mcpy( vreg_ptr( t, index * ebytes ), getmem( eaddr ), ebytes );
                        else
                            mcpy( getmem( eaddr ), vreg_ptr( t, index * ebytes ), ebytes );
                        offs += ebytes;
                        t = ( ( t + 1 ) % 32 );
                    }

                }

                if ( 31 != m )
                    offs = regs[ m ];

                if ( post_index )
                    regs[ n ] += offs;

                trace_vregs();
                break;
            }
            case 0x08: // LDAXRB <Wt>, [<Xn|SP>{, #0}]    ;    LDARB <Wt>, [<Xn|SP>{, #0}]    ;    STLXRB <Ws>, <Wt>, [<Xn|SP>{, #0}]    ;
                       // STXRB <Ws>, <Wt>, [<Xn|SP>{, #0}] ;  LDXRB <Wt>, [<Xn|SP>{, #0}]
            case 0x48: // LDAXRH <Wt>, [<Xn|SP>{, #0}]    ;    LDARH <Wt>, [<Xn|SP>{, #0}]    ;    STLXRH <Ws>, <Wt>, [<Xn|SP>{, #0}]    ;    STLRH <Wt>, [<Xn|SP>{, #0}]
                       // STXRH <Ws>, <Wt>, [<Xn|SP>{, #0}] ;  LDXRH <Wt>, [<Xn|SP>{, #0}]
            {
                uint64_t bit23 = opbit( 23 );
                uint64_t L = opbit( 22 );
                uint64_t bit21 = opbit( 21 );
                uint64_t s = opbits( 16, 5 );
                uint64_t t2 = opbits( 10, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t t = opbits( 0, 5 );
                bool is16 = opbit( 30 );

                if ( 0 != bit21 || 0x1f != t2 )
                    unhandled();

                if ( L )
                {
                    if ( 31 == t )
                        break;

                    if ( 0x1f != s )
                        unhandled();

                    if ( is16 )
                        regs[ t ] = getui16( regs[ n ] );
                    else
                        regs[ t ] = getui8( regs[ n ] );
                }
                else
                {
                    if ( !bit23 && 31 != s )
                        regs[ s ] = 0; // indicate the store succeeded for stlxrW and stxrW

                    if ( is16 )
                        setui16( regs[ n ], 0xffff & val_reg_or_zr( t ) );
                    else
                        setui8( regs[ n ], 0xff & val_reg_or_zr( t ) );
                }
                break;
            }
            case 0x1f: // fmadd, fnmadd, fmsub, fnmsub
            {
                uint64_t ftype = opbits( 22, 2 );
                uint64_t m = opbits( 16, 5 );
                uint64_t a = opbits( 10, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t subtract = opbit( 15 );
                uint64_t negate = opbit( 21 );

                if ( 0 == ftype ) // float
                {
                    // do math using doubles to match the behavior of arm64 hardware (test app ttypes.c validates this)

                    double product = do_fmul( vregs[ n ].getf( 0 ), vregs[ m ].getf( 0 ) );
                    if ( subtract )
                    {
                        if ( negate )
                            vregs[ d ].setf( 0, (float) do_fsub( product, vregs[ a ].getf( 0 ) ) );
                        else
                            vregs[ d ].setf( 0, (float) do_fsub( vregs[ a ].getf( 0 ), product ) );
                    }
                    else
                    {
                        if ( negate )
                            vregs[ d ].setf( 0, (float) do_fsub( -product, vregs[ a ].getf( 0 ) ) );
                        else
                            vregs[ d ].setf( 0, (float) do_fadd( product, vregs[ a ].getf( 0 ) ) );
                    }
                    memset( vreg_ptr( d, 4 ), 0, 12 );
                }
                else if ( 1 == ftype ) // double
                {
                    double product = do_fmul( vregs[ n ].getd( 0 ), vregs[ m ].getd( 0 ) );
                    if ( subtract )
                    {
                        if ( negate )
                            vregs[ d ].setd( 0, do_fsub( product, vregs[ a ].getd( 0 ) ) );
                        else
                            vregs[ d ].setd( 0, do_fsub( vregs[ a ].getd( 0 ), product ) );
                    }
                    else
                    {
                        if ( negate )
                            vregs[ d ].setd( 0, do_fsub( -product, vregs[ a ].getd( 0 ) ) );
                        else
                            vregs[ d ].setd( 0, do_fadd( product, vregs[ a ].getd( 0 ) ) );
                    }
                    memset( vreg_ptr( d, 8 ), 0, 8 );
                }
                else
                    unhandled();
                trace_vregs();
                break;
            }
            case 0x3c: // LDR <Bt>, [<Xn|SP>], #<simm>    ;    LDR <Bt>, [<Xn|SP>, #<simm>]!    ;    LDR <Qt>, [<Xn|SP>], #<simm>    ;     LDR <Qt>, [<Xn|SP>, #<simm>]!    ;    STUR <Bt>, [<Xn|SP>{, #<simm>}]
            case 0x3d: // LDR <Bt>, [<Xn|SP>{, #<pimm>}]  ;    LDR <Qt>, [<Xn|SP>{, #<pimm>}]
            case 0x7c: // LDR <Ht>, [<Xn|SP>], #<simm>    ;    LDR <Ht>, [<Xn|SP>, #<simm>]!
            case 0x7d: // LDR <Ht>, [<Xn|SP>{, #<pimm>}]
            case 0xbc:
            case 0xbd: // LDR <Dt>, [<Xn|SP>{, #<pimm>}]
            case 0xfc: // LDR <Dt>, [<Xn|SP>], #<simm>    ;    LDR <Dt>, [<Xn|SP>, #<simm>]!    ;    STR <Dt>, [<Xn|SP>], #<simm>    ;    STR <Dt>, [<Xn|SP>, #<simm>]!
            case 0xfd: // LDR <Dt>, [<Xn|SP>{, #<pimm>}]  ;    STR <Dt>, [<Xn|SP>{, #<pimm>}]
            {
                uint64_t bits11_10 = opbits( 10, 2 );
                uint64_t bit21 = opbit( 21 );
                bool unsignedOffset = ( 0xd == ( hi8 & 0xf ) );
                bool endsWithC = ( 0xc == ( hi8 & 0xf ) );
                bool preIndex = ( endsWithC && 3 == bits11_10 );
                bool postIndex = ( endsWithC && 1 == bits11_10 );
                bool signedUnscaledOffset = ( endsWithC && 0 == bits11_10 );
                bool shiftExtend = ( endsWithC && bit21 && 2 == bits11_10 );
                int64_t imm9 = sign_extend( opbits( 12, 9 ), 8 );
                uint64_t size = opbits( 30, 2 );
                uint64_t opc = opbits( 22, 2 );
                bool is_ldr = opbit( 22 );
                uint64_t t = opbits( 0, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t address = regs[ n ];
                uint64_t byte_len = 1ull << size;

                if ( ( is_ldr && 3 == opc ) || ( !is_ldr && 2 == opc ) )
                    byte_len = 16;

                //tracer.Trace( "hi8 %#x, str/ldr %d, preindex %d, postindex %d, signedUnscaledOffset %d, shiftExtend %d, imm9 %lld, n %llu, t %llu, byte_len %llu\n",
                //              hi8, is_ldr, preIndex, postIndex, signedUnscaledOffset, shiftExtend, imm9, n, t, byte_len );

                if ( preIndex )
                {
                     regs[ n ] += imm9;
                     address = regs[ n ];
                }
                else if ( unsignedOffset )
                {
                    uint64_t imm12 = opbits( 10, 12 );
                    address += ( imm12 * byte_len );
                }
                else if ( signedUnscaledOffset )
                    address += imm9;
                else if ( shiftExtend )
                {
                    uint64_t option = opbits( 13, 3 );
                    uint64_t m = opbits( 16, 5 );
                    uint64_t shift = 0;
                    uint64_t S = opbit( 12 );
                    //tracer.Trace( "option %#llx, S %llu, size %llu, opc %#llx\n", option, S, size, opc );
                    if ( 0 != S )
                    {
                        if ( is_ldr )
                        {
                            if ( 0 == size )
                            {
                               if ( 3 == opc )
                                   shift = 4;
                               else if ( 1 != opc )
                                   unhandled();
                            }
                            else if ( 1 == opc && size <= 3 )
                                shift = size;
                            else
                                unhandled();
                        }
                        else
                        {
                            if ( 0 == size )
                            {
                               if ( 2 == opc )
                                   shift = 4;
                               else if ( 0 != opc )
                                   unhandled();
                            }
                            else if ( 0 == opc && size <= 3 )
                                shift = size;
                            else
                                unhandled();
                        }
                    }
                    int64_t offset = extend_reg( m, option, shift );
                    address += offset;
                }
                else if ( !postIndex )
                    unhandled();

                if ( is_ldr )
                {
                    zero_vreg( t );
                    mcpy( vreg_ptr( t, 0 ), getmem( address ), byte_len );
                }
                else
                    mcpy( getmem( address ), vreg_ptr( t, 0 ), byte_len );

                if ( postIndex )
                     regs[ n ] += imm9;

                trace_vregs();
                break;
            }
            case 0x2c: // STP <St1>, <St2>, [<Xn|SP>], #<imm>     ;    LDP <St1>, <St2>,
            case 0x6c: // STP <Dt1>, <Dt2>, [<Xn|SP>], #<imm>     ;    LDP <Dt1>, <Dt2>, [<Xn|SP>], #<imm>
            case 0xac: // STP <Qt1>, <Qt2>, [<Xn|SP>], #<imm>          LDP <Qt1>, <Qt2>, [<Xn|SP>], #<imm>
            case 0x2d: // STP <St1>, <St2>, [<Xn|SP>, #<imm>]!    ;    STP <St1>, <St2>, [<Xn|SP>{, #<imm>}]    ;    LDP <St1>, <St2>, [<Xn|SP>, #<imm>]!    ;    LDP <St1>, <St2>, [<Xn|SP>{, #<imm>}]
            case 0x6d: // STP <Dt1>, <Dt2>, [<Xn|SP>, #<imm>]!    ;    STP <Dt1>, <Dt2>, [<Xn|SP>{, #<imm>}]    ;    LDP <Dt1>, <Dt2>, [<Xn|SP>, #<imm>]!    ;    LDP <Dt1>, <Dt2>, [<Xn|SP>{, #<imm>}]
            case 0xad: // STP <Qt1>, <Qt2>, [<Xn|SP>, #<imm>]!    ;    STP <Qt1>, <Qt2>, [<Xn|SP>{, #<imm>}]    ;    LDP <Qt1>, <Qt2>, [<Xn|SP>, #<imm>]!    ;    LDP <Qt1>, <Qt2>, [<Xn|SP>{, #<imm>}]
            {
                uint64_t opc = opbits( 30, 2 );
                uint64_t imm7 = opbits( 15, 7 );
                uint64_t t2 = opbits( 10, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t t1 = opbits( 0, 5 );
                uint64_t L = opbit( 22 );
                uint64_t bit23 = opbit( 23 );

                bool preIndex = ( ( 0xd == ( hi8 & 0xf ) ) && bit23 );
                bool postIndex = ( ( 0xc == ( hi8 & 0xf ) ) && bit23 );
                bool signedOffset = ( ( 0xd == ( hi8 & 0xf ) ) && !bit23 );

                uint64_t scale = 2 + opc;
                int64_t offset = sign_extend( imm7, 6 ) << scale;
                uint64_t address = regs[ n ];
                uint64_t byte_len = 4ull << opc;

                if ( preIndex || signedOffset )
                    address += offset;

                if ( 1 == L ) // ldp
                {
                    zero_vreg( t1 );
                    zero_vreg( t2 );
                    mcpy( vreg_ptr( t1, 0 ), getmem( address ), byte_len );
                    mcpy( vreg_ptr( t2, 0 ), getmem( address + byte_len ), byte_len );
                }
                else // stp
                {
                    mcpy( getmem( address ), vreg_ptr( t1, 0 ), byte_len );
                    mcpy( getmem( address + byte_len ), vreg_ptr( t2, 0 ), byte_len );
                }

                if ( postIndex )
                    address += offset;

                if ( !signedOffset )
                    regs[ n ] = address;

                trace_vregs();
                break;
            }
            case 0x0f: case 0x2f: case 0x4f: case 0x6f: case 0x7f:
                // BIC <Vd>.<T>, #<imm8>{, LSL #<amount>}    ;    MOVI <Vd>.<T>, #<imm8>{, LSL #0}    ;    MVNI <Vd>.<T>, #<imm8>, MSL #<amount>
                // USHR <Vd>.<T>, <Vn>.<T>, #<shift>         ;    FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                // FMOV <Vd>.<T>, #<imm>                     ;    FMOV <Vd>.<T>, #<imm>               ;    FMOV <Vd>.2D, #<imm>
                // USHLL{2} <Vd>.<Ta>, <Vn>.<Tb>, #<shift>   ;    SHRN{2} <Vd>.<Tb>, <Vn>.<Ta>, #<shift>  ;   SSHLL{2} <Vd>.<Ta>, <Vn>.<Tb>, #<shift>
                // FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>] ;  SSHR <Vd>.<T>, <Vn>.<T>, #<shift>   ;    SHL <Vd>.<T>, <Vn>.<T>, #<shift>
                // MUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>] ;   URSRA <Vd>.<T>, <Vn>.<T>, #<shift>  ;    USRA <Vd>.<T>, <Vn>.<T>, #<shift>
                // ORR <Vd>.<T>, #<imm8>{, LSL #<amount>}    ;    MOVI <Vd>.<T>, #<imm8>, MSL #<amount>
            {
                uint64_t cmode = opbits( 12, 4 );
                uint64_t abc = opbits( 16, 3 );
                uint64_t defgh = opbits( 5, 5 );
                uint64_t val = ( abc << 5 ) | defgh;
                uint64_t Q = opbit( 30 );
                uint64_t bit29 = opbit( 29 );
                uint64_t bit10 = opbit( 10 );
                uint64_t bit11 = opbit( 11 );
                uint64_t bit12 = opbit( 12 );
                uint64_t bit13 = opbit( 13 );
                uint64_t bit14 = opbit( 14 );
                uint64_t bit15 = opbit( 15 );
                uint64_t bit23 = opbit( 23 );
                uint64_t d = opbits( 0, 5 );
                uint64_t bits23_19 = opbits( 19, 5 );
                uint64_t imm = adv_simd_expand_imm( bit29, cmode, val );

                if ( 0 == bits23_19 )
                {
                    if ( ( 0x4f == hi8 || 0x0f == hi8 ) && ( 0xc == ( cmode & 0xe ) ) && !bit11 && bit10 ) // MOVI <Vd>.<T>, #<imm8>, MSL #<amount>
                    {
                        vregs[ d ].set64( 0, imm );
                        if ( Q )
                            vregs[ d ].set64( 1, imm );
                        else
                            vregs[ d ].set64( 1, 0 );
                    }
                    else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && ( 9 == ( 0xd & cmode ) || 1 == ( 9 & cmode ) ) && bit12 && !bit11 && bit10 ) // ORR <Vd>.<T>, #<imm8>{, LSL #<amount>}
                    {
                        if ( 9 == ( 0xd & cmode ) ) // 16-bit variant
                        {
                            uint64_t elements = Q ? 8 : 4;
                            for ( uint64_t e = 0; e < elements; e++ )
                                vregs[ d ].set16( e, vregs[ d ].get16( e ) | (uint16_t) imm );
                        }
                        else if ( 1 == ( 9 & cmode ) ) // 32-bit variant
                        {
                            uint64_t elements = Q ? 4 : 2;
                            for ( uint64_t e = 0; e < elements; e++ )
                                vregs[ d ].set32( e, vregs[ d ].get32( e ) | (uint32_t) imm );
                        }
                        else
                            unhandled();

                    }
                    else if ( ( 0x2f == hi8 || 0x6f == hi8 ) && !bit11 && bit10 && // MOVI <Vd>.<T>, #<imm8>{, LSL #0}    ;    MVNI <Vd>.<T>, #<imm8>, MSL #<amount>
                              ( ( 8 == ( cmode & 0xd ) ) || ( 0 == ( cmode & 9 ) ) || ( 0xc == ( cmode & 0xe ) ) ) ) // mvni
                    {
                        if ( 8 == ( cmode & 0xd ) ) // 16-bit shifted immediate
                        {
                            uint64_t amount = get_bit( cmode, 1 ) * 8;
                            val <<= amount;
                            uint16_t invval = (uint16_t) ~val;
                            for ( uint64_t o = 0; o < ( Q ? 8 : 4 ); o++ )
                                vregs[ d ].set16( o, invval );
                        }
                        else if ( 0 == ( cmode & 9 ) ) // 32-bit shifted immediate
                        {
                            uint64_t amount = get_bits( cmode, 1, 2 ) * 8;
                            val <<= amount;
                            uint32_t invval = (uint32_t) ~val;
                            for ( uint64_t o = 0; o < ( Q ? 4 : 2 ); o++ )
                                vregs[ d ].set32( o, invval );
                        }
                        else if ( 0xc == ( cmode & 0xe ) ) // 32-bit shifting ones
                        {
                            uint64_t invimm = (uint64_t) ~imm;
                            vregs[ d ].set64( 0, invimm );
                            if ( Q )
                                vregs[ d ].set64( 1, invimm );
                        }
                        else
                            unhandled();
                    }
                    else if ( !bit12 || ( 0xc == ( cmode & 0xe ) ) ) // movi
                    {
                        if ( !bit29)
                        {
                            if ( 0xe == cmode ) // 64-bit
                            {
                                vregs[ d ].set64( 0, imm );
                                vregs[ d ].set64( 1, Q ? imm : 0 );
                            }
                            else if ( 8 == ( cmode & 0xd ) ) // 16-bit shifted immediate
                            {
                                uint64_t amount = ( cmode & 2 ) ? 8 : 0;
                                val <<= amount;
                                zero_vreg( d );
                                for ( uint64_t o = 0; o < ( Q ? 8 : 4 ); o++ )
                                    vregs[ d ].set16( o, (uint16_t) val );
                            }
                            else if ( 0 == ( cmode & 9 ) ) // 32-bit shifted immediate
                            {
                                uint64_t amount = ( 8 * ( ( cmode >> 1 ) & 3 ) );
                                val <<= amount;
                                val = replicate_bytes( val, 4 );
                                vregs[ d ].set64( 0, val );
                                vregs[ d ].set64( 1, Q ? val : 0 );
                            }
                            else if ( 0xa == ( cmode & 0xe ) )
                            {
                                //uint64_t amount = ( cmode & 1 ) ? 16 : 8;
                                unhandled();
                            }
                            else
                                unhandled();
                        }
                        else
                        {
                            uint64_t a = opbit( 18 );
                            uint64_t b = opbit( 17 );
                            uint64_t c = opbit( 16 );
                            uint64_t dbit = opbit( 9 );
                            uint64_t e = opbit( 8 );
                            uint64_t f = opbit( 7 );
                            uint64_t g = opbit( 6 );
                            uint64_t h = opbit( 5 );

                            imm = a ? ( 0xffull << 56 ) : 0;
                            imm |= b ? ( 0xffull << 48 ) : 0;
                            imm |= c ? ( 0xffull << 40 ) : 0;
                            imm |= dbit ? ( 0xffull << 32 ) : 0;
                            imm |= e ? ( 0xffull << 24 ) : 0;
                            imm |= f ? ( 0xffull << 16 ) : 0;
                            imm |= g ? ( 0xffull << 8 ) : 0;
                            imm |= h ? 0xffull : 0;

                            if ( ( 0 == Q ) && ( cmode == 0xe ) )
                                vregs[ d ].set64( 0, imm );
                            else if ( ( 1 == Q ) && ( cmode == 0xe ) )
                            {
                                vregs[ d ].set64( 0, imm );
                                vregs[ d ].set64( 1, imm );
                            }
                            else
                                unhandled();
                        }
                    }
                    else if ( ( 0x6f == hi8 || 0x4f == hi8 || 0x2f == hi8 || 0x0f == hi8 ) && 0xf == cmode && !bit11 && bit10 ) // fmov single and double precision immediate
                    {
                        zero_vreg( d );
                        if ( bit29 )
                        {
                            vregs[ d ].set64( 0, imm );
                            if ( Q )
                                vregs[ d ].set64( 1, imm );
                        }
                        else
                        {
                            vregs[ d ].set32( 0, (uint32_t) imm );
                            vregs[ d ].set32( 1, (uint32_t) imm );
                            if ( Q )
                            {
                                vregs[ d ].set32( 2, (uint32_t) imm );
                                vregs[ d ].set32( 3, (uint32_t) imm );
                            }
                        }
                    }
                    else if ( !bit29 ) // BIC register
                    {
                        unhandled();
                    }
                    else if ( bit29 && bit12 ) // BIC immediate
                    {
                        uint64_t notimm = ~imm;

                        if ( 9 == ( cmode & 0xd ) ) // 16-bit mode
                        {
                            uint64_t limit = ( 0 == Q ) ? 4 : 8;
                            for ( uint64_t i = 0; i < limit; i++ )
                                vregs[ d ].set16( i, vregs[ d ].get16( i ) & (uint16_t) notimm );
                        }
                        else if ( 1 == ( cmode & 1 ) ) // 32-bit mode
                        {
                            uint64_t limit = ( 0 == Q ) ? 2 : 4;
                            for ( uint64_t i = 0; i < limit; i++ )
                                vregs[ d ].set32( i, vregs[ d ].get32( i ) & (uint32_t) notimm );
                        }
                        else
                            unhandled();
                    }
                }
                else // USHR, USHLL, SHRN, SHRN2, etc
                {
                    uint64_t opcode = opbits( 12, 4 );
                    uint64_t bits23_22 = opbits( 22, 2 );
                    uint64_t bits15_12 = opbits( 12, 4 );
                    uint64_t immh = opbits( 19, 4 );

                    if ( ( 0x2f == hi8 || 0x6f == hi8 ) && !bit23 && 0 != immh && 1 == bits15_12 && !bit11 && bit10 ) // USRA <Vd>.<T>, <Vn>.<T>, #<shift>
                    {
                        uint64_t immb = opbits( 16, 3 );
                        uint64_t n = opbits( 5, 5 );
                        uint64_t esize = 8ull << highest_set_bit_nz( immh );
                        uint64_t ebytes = esize / 8;
                        assert( 1 == count_bits( ebytes ) );
                        uint64_t shift = ( esize * 2 ) - ( ( immh << 3 ) | immb );
                        uint64_t datasize = 64ull << Q;
                        uint64_t elements = datasize / esize;

                        vec16_t & dref = vregs[ d ];
                        vec16_t & nref = vregs[ n ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                dref.set8( e, dref.get8( e ) + ( nref.get8( e ) >> shift ) );
                            else if ( 2 == ebytes )
                                dref.set16( e, dref.get16( e ) + ( nref.get16( e ) >> shift ) );
                            else if ( 4 == ebytes )
                                dref.set32( e, dref.get32( e ) + ( nref.get32( e ) >> shift ) );
                            else
                                dref.set64( e, dref.get64( e ) + ( nref.get64( e ) >> shift ) );
                        }
                    }
                    else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && !bit23 && 0 != immh && 1 == bits15_12 && !bit11 && bit10 ) // SSRA <Vd>.<T>, <Vn>.<T>, #<shift>
                    {
                        uint64_t immb = opbits( 16, 3 );
                        uint64_t n = opbits( 5, 5 );
                        uint64_t esize = 8ull << highest_set_bit_nz( immh );
                        uint64_t ebytes = esize / 8;
                        assert( 1 == count_bits( ebytes ) );
                        uint64_t shift = ( esize * 2 ) - ( ( immh << 3 ) | immb );
                        uint64_t datasize = 64ull << Q;
                        uint64_t elements = datasize / esize;
                        vec16_t & dref = vregs[ d ];
                        vec16_t & nref = vregs[ n ];
                        //tracer.Trace( "ssra esize %llu, shift %llu, elements %llu\n", esize, shift, elements );
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                dref.set8( e, dref.get8( e ) + ( ( (int8_t) nref.get8( e ) ) >> shift ) );
                            else if ( 2 == ebytes )
                                dref.set16( e, dref.get16( e ) + ( ( (int16_t) nref.get16( e ) ) >> shift ) );
                            else if ( 4 == ebytes )
                                dref.set32( e, dref.get32( e ) + ( ( (int32_t) nref.get32( e ) ) >> shift ) );
                            else
                                dref.set64( e, dref.get64( e ) + ( ( (int64_t) nref.get64( e ) ) >> shift ) );
                        }
                    }
                    else if ( ( ( 1 == bits23_22 || 2 == bits23_22 ) && !bit10 ) &&
                         ( ( ( 0x4f == hi8 || 0x0f == hi8 ) && 8 == bits15_12 ) ||    // MUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                           ( ( 0x2f == hi8 || 0x6f == hi8 ) && 0 == bits15_12 ) ) )   // MLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                    {
                        uint64_t size = bits23_22;
                        uint64_t n = opbits( 5, 5 );
                        uint64_t L = opbit( 21 );
                        uint64_t M = opbit( 20 );
                        uint64_t H = opbit( 11 );
                        uint64_t index = 0;
                        uint64_t rmhi = 0;
                        uint64_t indx_val_size = 0;
                        if ( 1 == size )
                        {
                            indx_val_size = 2;
                            index = ( H << 2 ) | ( L << 1 ) | M;
                            rmhi = 0;
                        }
                        else if ( 2 == size )
                        {
                            indx_val_size = 4;
                            index = ( H << 1 ) | L;
                            rmhi = M;
                        }
                        else
                            unhandled();

                        uint64_t m = ( rmhi << 4 ) | opbits( 16, 4 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t datasize = 64ull << Q;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        uint64_t element2 = 0;
                        if ( 2 == indx_val_size )
                            element2 = vregs[ m ].get16( index );
                        else
                            element2 = vregs[ m ].get32( index );
                        bool accumulate = ( 0x2f == hi8 || 0x6f == hi8 );
                        vec16_t & vn = vregs[ n ];
                        vec16_t & vd = vregs[ d ];

                        if ( 1 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set8( e, (uint8_t) ( ( vn.get8( e ) * element2 ) + ( accumulate ? vd.get8( e ) : 0 ) ) );
                        else if ( 2 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set16( e, (uint16_t) ( ( vn.get16( e ) * element2 ) + ( accumulate ? vd.get16( e ) : 0 ) ) );
                        else if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set32( e, (uint32_t) ( ( vn.get32( e ) * element2 ) + ( accumulate ? vd.get32( e ) : 0 ) ) );
                        else if ( 8 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set64( e, (uint64_t) ( ( vn.get64( e ) * element2 ) + ( accumulate ? vd.get64( e ) : 0 ) ) );
                        else
                            unhandled();

                        vregs[ d ] = target;
                    }
                    else if ( ( 0x4f == hi8 || 0x0f == hi8 ) && !bit23 && 5 == opcode && !bit11 && bit10 ) // SHL <Vd>.<T>, <Vn>.<T>, #<shift>
                    {
                        uint64_t n = opbits( 5, 5 );
                        uint64_t immb = opbits( 16, 3 );
                        uint64_t esize = 8ull << highest_set_bit_nz( immh );
                        uint64_t ebytes = esize / 8;
                        assert( 1 == count_bits( ebytes ) );
                        uint64_t shift = ( ( immh << 3 ) | immb ) - esize;
                        uint64_t datasize = 64ull << Q;
                        uint64_t elements = datasize / esize;
                        vec16_t & nref = vregs[ n ];
                        vec16_t & dref = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                dref.set8( e, ( nref.get8( e ) << shift ) );
                            else if ( 2 == ebytes )
                                dref.set16( e, ( nref.get16( e ) << shift ) );
                            else if ( 4 == ebytes )
                                dref.set32( e, ( nref.get32( e ) << shift ) );
                            else
                                dref.set64( e, ( nref.get64( e ) << shift ) );
                        }
                    }
                    else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && !bit23 && 0 == opcode && !bit11 && bit10 ) // SSHR <Vd>.<T>, <Vn>.<T>, #<shift>
                    {
                        uint64_t n = opbits( 5, 5 );
                        uint64_t immb = opbits( 16, 3 );
                        uint64_t esize = 8ull << highest_set_bit_nz( immh );
                        uint64_t ebytes = esize / 8;
                        assert( 1 == count_bits( ebytes ) );
                        uint64_t datasize = 64ull << Q;
                        uint64_t elements = datasize / esize;
                        uint64_t shift = ( esize * 2 ) - ( ( immh << 3 ) | immb );
                        vec16_t target;

                        if ( 1 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set8( e, ( (int8_t) vregs[ n ].get8( e ) ) >> shift );
                        else if ( 2 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set16( e, ( (int16_t) vregs[ n ].get16( e ) ) >> shift );
                        else if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set32( e, ( (int32_t) vregs[ n ].get32( e ) ) >> shift );
                        else if ( 8 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set64( e, ( (int64_t) vregs[ n ].get64( e ) ) >> shift );

                        vregs[ d ] = target;
                    }
                    else if ( ( 0x4f == hi8 || 0x0f == hi8 ) && bit23 && 1 == opcode && !bit10 ) // FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
                    {
                        uint64_t n = opbits( 5, 5 );
                        uint64_t m = opbits( 16, 5 );
                        uint64_t sz = opbit( 22 );
                        uint64_t L = opbit( 21 );
                        uint64_t H = opbit( 11 );
                        uint64_t szL = ( sz << 1 ) | L;
                        uint64_t index = ( 0 == sz ) ? ( ( H << 1 ) | L ) : ( 2 == szL ) ? H : 0;
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t datasize = 64ull << Q;
                        uint64_t elements = datasize / esize;
                        vec16_t target = vregs[ d ];
                        vec16_t & vn = vregs[ n ];
                        // tracer.Trace( "elements %llu, esize %llu, idxsize %llu, datasize %llu, d %llu, n %llu, m %llu, index %llu\n", elements, esize, idxsize, datasize, d, n, m, index );

                        if ( 8 == ebytes )
                        {
                            double element2 = vregs[ m ].getd( index );
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.setd( e, do_fadd( target.getd( e ), do_fmul( element2, vn.getd( e ) ) ) );
                        }
                        else if ( 4 == ebytes )
                        {
                            double element2 = (double) vregs[ m ].getf( index );
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.setf( e, (float) do_fadd( target.getf( e ), do_fmul( element2, vn.getf( e ) ) ) );
                        }
                        else
                            unhandled() ;

                        vregs[ d ] = target;
                    }
                    else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && !bit23 && 0 != bits23_19 && 0xa == opcode && !bit11 && bit10 ) // SSHLL{2} <Vd>.<Ta>, <Vn>.<Tb>, #<shift>
                    {
                        uint64_t n = opbits( 5, 5 );
                        uint64_t immb = opbits( 16, 3 );
                        uint64_t esize = 8ull << highest_set_bit_nz( immh & 0x7 );
                        uint64_t ebytes = esize / 8;
                        assert( 1 == count_bits( ebytes ) );
                        uint64_t shift = ( ( immh << 3 ) | immb ) - esize;
                        uint64_t datasize = 64;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        //tracer.Trace( "sshl{2} shift %llu, ebytes %llu, elements %llu\n", shift, ebytes, elements );

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, sign_extend16( vregs[ n ].get8( e + ( Q ? 8 : 0 ) ), 7 ) << shift );
                            else if ( 2 == ebytes )
                                target.set32( e, sign_extend32( vregs[ n ].get16( e + ( Q ? 4 : 0 ) ), 15 ) << shift );
                            else if ( 4 == ebytes )
                                target.set64( e, sign_extend( vregs[ n ].get32( e + ( Q ? 2 : 0 ) ), 31 ) << shift );
                            else
                                unhandled();
                        }

                        vregs[ d ] = target;
                    }
                    else if ( ( 0x0f == hi8 || 0x4f == hi8 ) && !bit23 && 0 != bits23_19 && 8 == opcode && !bit11 && bit10 ) // SHRN{2} <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
                    {
                        uint64_t n = opbits( 5, 5 );
                        uint64_t immb = opbits( 16, 3 );
                        uint64_t esize = 8ull << highest_set_bit_nz( immh & 0x7 );
                        uint64_t ebytes = esize / 8;
                        assert( ebytes <= 4 );
                        assert( 1 == count_bits( ebytes ) );
                        uint64_t datasize = 64;
                        uint64_t part = Q;
                        uint64_t elements = datasize / esize;
                        uint64_t shift = ( 2 * esize ) - ( ( immh << 3 ) | immb );
                        vec16_t target;

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set8( e, (uint8_t) ( vregs[ n ].get16( e ) >> shift ) );
                            else if ( 2 == ebytes )
                                target.set16( e, (uint16_t) ( vregs[ n ].get32( e ) >> shift ) );
                            else if ( 4 == ebytes )
                                target.set32( e, (uint32_t) ( vregs[ n ].get64( e ) >> shift ) );
                            else
                                unhandled();
                        }

                        if ( part )
                            vregs[ d ].set64( 1, target.get64( 0 ) );
                        else
                        {
                            vregs[ d ].set64( 0, target.get64( 0 ) );
                            vregs[ d ].set64( 1, 0 );
                        }
                    }
                    else if ( ( 0x2f == hi8 || 0x6f == hi8 ) && !bit23 && 0 != bits23_19 && ( 0xa == opcode ) && !bit11 && bit10 ) // USHLL{2} <Vd>.<Ta>, <Vn>.<Tb>, #<shift>
                    {
                        uint64_t n = opbits( 5, 5 );
                        uint64_t immb = opbits( 16, 3 );
                        uint64_t esize = 8ull << highest_set_bit_nz( immh & 0x7 );
                        uint64_t ebytes = esize / 8;
                        assert( ebytes <= 4 );
                        assert( 1 == count_bits( ebytes ) );
                        uint64_t datasize = 64;
                        uint64_t elements = datasize / esize;
                        uint64_t shift = ( ( immh << 3 ) | immb ) - esize;
                        vec16_t target;

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, ( (uint16_t) vregs[ n ].get8( e + ( Q ? 8 : 0 ) ) ) << shift );
                            else if ( 2 == ebytes )
                                target.set32( e, ( (uint32_t) vregs[ n ].get16( e + ( Q ? 4 : 0 ) ) ) << shift );
                            else if ( 4 == ebytes )
                                target.set64( e, ( (uint64_t) vregs[ n ].get32( e + ( Q ? 2 : 0 ) ) ) << shift );
                            else
                                unhandled();
                        }

                        vregs[ d ] = target;
                    }
                    else if ( ( 0x2f == hi8 || 0x7f == hi8 || 0x6f == hi8 ) && !bit23 && !bit15 && !bit14 && !bit13 && !bit12 && !bit11 && bit10 ) // USHR
                    {
                        uint64_t n = opbits( 5, 5 );
                        uint64_t immb = opbits( 16, 3 );
                        uint64_t esize = 8ull << highest_set_bit_nz( immh );
                        if ( 0x7f == hi8 )
                            esize = 8ull << 3;
                        uint64_t ebytes = esize / 8;
                        assert( 1 == count_bits( ebytes ) );
                        uint64_t datasize = 64ull << Q;
                        if ( 0x7f == hi8 )
                            datasize = esize;
                        uint64_t elements = datasize / esize;
                        if ( 0x7f == hi8 )
                            elements = 1;
                        uint64_t shift = ( esize * 2 ) - ( ( immh << 3 ) | immb );
                        vec16_t target;

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set8( e, vregs[ n ].get8( e ) >> shift );
                            else if ( 2 == ebytes )
                                target.set16( e, vregs[ n ].get16( e ) >> shift );
                            else if ( 4 == ebytes )
                                target.set32( e, vregs[ n ].get32( e ) >> shift );
                            else if ( 8 == ebytes )
                                target.set64( e, vregs[ n ].get64( e ) >> shift );
                        }

                        vregs[ d ] = target;
                    }
                    else if ( bit23 && !bit10 && 9 == opcode ) // FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]. Vector, single-precision and double-precision
                    {
                        uint64_t n = opbits( 5, 5 );
                        uint64_t m = opbits( 16, 5 );
                        uint64_t sz = opbit( 22 );
                        uint64_t L = opbit( 21 );
                        uint64_t H = opbit( 11 );

                        uint64_t index = ( !sz ) ? ( ( H << 1 ) | L ) : H;
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t datasize = 64ull << Q;
                        uint64_t elements = datasize / esize;

                        //tracer.Trace( "index: %llu, esize %llu, ebytes %llu, datasize %llu, elements %llu\n", index, esize, ebytes, datasize, elements );
                        vec16_t & vn = vregs[ n ];
                        vec16_t & vd = vregs[ d ];

                        if ( 8 == ebytes )
                        {
                            double mval = vregs[ m ].getd( index );
                            for ( uint64_t e = 0; e < elements; e++ )
                                vd.setd( e, do_fmul( vn.getd( e ), mval ) );
                        }
                        else if ( 4 == ebytes )
                        {
                            float mval = vregs[ m ].getf( index );
                            for ( uint64_t e = 0; e < elements; e++ )
                                vd.setf( e, (float) do_fmul( vn.getf( e ), mval ) );
                        }
                        else
                            unhandled();
                    }
                    else
                        unhandled();
                }

                trace_vregs();
                break;
            }
            case 0x5a: // REV <Wd>, <Wn>  ;  CSINV <Wd>, <Wn>, <Wm>, <cond>  ;  RBIT <Wd>, <Wn>  ;  CLZ <Wd>, <Wn>  ;  CSNEG <Wd>, <Wn>, <Wm>, <cond>  ;  SBC <Wd>, <Wn>, <Wm> ; REV16 <Wd>, <Wn>
            case 0xda: // REV <Xd>, <Xn>  ;  CSINV <Xd>, <Xn>, <Xm>, <cond>  ;  RBIT <Xd>, <Xn>  ;  CLZ <Xd>, <Xn>  ;  CSNEG <Xd>, <Xn>, <Xm>, <cond>  ;  SBC <Xd>, <Xn>, <Xm> ; REV16 <Xd>, <Xn>
            {
                uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
                uint64_t opc = opbits( 10, 2 ); // 2 or 3 for container size
                uint64_t data_size = ( 32ull << opbit( 31 ) );
                uint64_t container_size = ( 8ull << opc );
                uint64_t containers = data_size / container_size;
                uint64_t bits23_21 = opbits( 21, 3 );
                uint64_t bits23_10 = opbits( 10, 14 );
                uint64_t bits15_10 = opbits( 10, 6 );
                uint64_t bit11 = opbit( 11 );
                uint64_t bit10 = opbit( 10 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t result = 0;
                uint64_t nval = val_reg_or_zr( n );

                if ( 0x3001 == bits23_10 ) // rev16
                {
                    for ( uint64_t c = 0; c < containers; c++ )
                    {
                        uint64_t container = get_elem_bits( nval, c, container_size );
                        result |= flip_endian16( (uint16_t) get_elem_bits( container, c, container_size ) );
                    }
                }
                else if ( 4 == bits23_21 ) // csinv / csneg
                {
                    if ( bit11 )
                        unhandled();
                    uint64_t cond = opbits( 12, 4 );
                    if ( check_conditional( cond ) )
                        result = val_reg_or_zr( n );
                    else
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t mval = val_reg_or_zr( m );
                        if ( bit10 ) // csneg
                            result = ( (uint64_t) ( - (int64_t) mval ) ); // works for !xregs 32-bit result as well
                        else // csinv
                            result = ~ mval;
                    }
                }
                else if ( 6 == bits23_21 )
                {
                    if ( 0 == bits15_10 ) // rbit
                    {
                        uint64_t limit = xregs ? 64 : 32;
                        for ( uint64_t bit = 0; bit < limit; bit++ )
                        {
                            result <<= 1;
                            if ( nval & 1 )
                                result |= 1;
                            nval >>= 1;
                        }
                    }
                    else if ( 2 == bits15_10 || 3 == bits15_10 ) // rev
                    {
                        for ( uint64_t c = 0; c < containers; c++ )
                        {
                            uint64_t container = get_elem_bits( nval, c, container_size );
                            if ( 32 == container_size )
                                result |= get_elem_bits( flip_endian32( (uint32_t) container ), c, container_size );
                            else
                                result |= get_elem_bits( flip_endian64( container ), c, container_size );
                        }
                    }
                    else if ( 4 == bits15_10 ) // clz
                    {
                        if ( ! xregs )
                            nval = (uint32_t) nval;
                        while ( nval )
                        {
                            result++;
                            nval >>= 1;
                        }
                        if ( xregs )
                            result = 64 - result;
                        else
                            result = 32 - result;
                    }
                    else
                        unhandled();
                }
                else if ( 0 == bits23_21 )
                {
                    if ( 0 == bits15_10 ) // sbc
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t mval = val_reg_or_zr( m );

                        if ( xregs )
                            result = add_with_carry64( nval, ~mval, fC, false );
                        else
                            result = add_with_carry32( (uint32_t) nval, (uint32_t) ( ~ mval ), fC, false );
                    }
                    else
                        unhandled();
                }
                else
                    unhandled();

                if ( 31 == d )
                    break;
                if ( !xregs )
                    result = (uint32_t) result;
                regs[ d ] = result;
                break;
            }
            case 0x14: case 0x15: case 0x16: case 0x17: // b label
            {
                int64_t imm26 = opbits( 0, 26 );
                imm26 <<= 2;
                imm26 = sign_extend( imm26, 27 );
                pc += imm26;
                continue;
            }
            case 0x1a: // CSEL <Wd>, <Wn>, <Wm>, <cond>    ;    SDIV <Wd>, <Wn>, <Wm>    ;    UDIV <Wd>, <Wn>, <Wm>    ;    CSINC <Wd>, <Wn>, <Wm>, <cond>
                       // LSRV <Wd>, <Wn>, <Wm>            ;    LSLV <Wd>, <Wn>, <Wm>    ;    ADC <Wd>, <Wn>, <Wm>     ;    ASRV <Wd>, <Wn>, <Wm>
                       // RORV <Wd>, <Wn>, <Wm>
            case 0x9a: // CSEL <Xd>, <Xn>, <Xm>, <cond>    ;    SDIV <Xd>, <Xn>, <Xm>    ;    UDIV <Xd>, <Xn>, <Xm>    ;    CSINC <Xd>, <Xn>, <Xm>, <cond>
                       // LSRV <Xd>, <Xn>, <Xm>            ;    LSLV <Xd>, <Xn>, <Xm>    ;    ADC <Xd>, <Xn>, <Xm>     ;    ASRV <Xd>, <Xn>, <Xm>
                       // RORV <Xd>, <Xn>, <Xm>
            {
                uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
                uint64_t bits11_10 = opbits( 10, 2 );
                uint64_t d = opbits( 0, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t m = opbits( 16, 5 );
                uint64_t bits15_12 = opbits( 12, 4 );
                uint64_t bits23_21 = opbits( 21, 3 );
                if ( 31 == d )
                    break;

                uint64_t mval = val_reg_or_zr( m );
                uint64_t nval = val_reg_or_zr( n );

                if ( 0 == bits11_10 && 4 == bits23_21 ) // CSEL
                {
                    uint64_t cond = opbits( 12, 4 );
                    if ( check_conditional( cond ) )
                        regs[ d ] = nval;
                    else
                        regs[ d ] = mval;
                }
                else if ( 1 == bits11_10 && 4 == bits23_21 ) // CSINC <Xd>, XZR, XZR, <cond>
                {
                    uint64_t cond = opbits( 12, 4 );
                    if ( check_conditional( cond ) )
                        regs[ d ] = nval;
                    else
                        regs[ d ] = 1 + mval;
                }
                else if ( 2 == bits11_10 && 6 == bits23_21 && 2 == bits15_12 ) // ASRV <Xd>, <Xn>, <Xm>
                {
                    uint64_t shift = mval;
                    uint64_t result = 0;
                    if ( xregs )
                    {
                        shift = shift & 0x3f;
                        result = ( ( (int64_t) nval ) >> shift );
                    }
                    else
                    {
                        shift = ( (uint32_t) shift ) & 0x1f;
                        result = (uint32_t) ( ( (int32_t) nval ) >> shift );
                    }

                    regs[ d ] = result;
                }
                else if ( 2 == bits11_10 && 6 == bits23_21 && 0 == bits15_12 ) // UDIV <Xd>, <Xn>, <Xm>
                {
                    if ( xregs )
                        regs[ d ] = ( 0 == mval ) ? 0 : ( nval / mval );
                    else
                        regs[ d ] = ( (uint32_t) ( ( 0 == mval ) ? 0 : ( (uint32_t) nval / (uint32_t) mval ) ) );
                }
                else if ( 3 == bits11_10 && 6 == bits23_21 && 0 == bits15_12 ) // SDIV <Xd>, <Xn>, <Xm>
                {
                    if ( xregs )
                        regs[ d ] = ( 0 == mval ) ? 0 : ( (int64_t) nval / (int64_t) mval );
                    else
                        regs[ d ] = ( ( 0 == mval ) ? 0 : ( (int32_t) ( (uint32_t) nval ) / (int32_t) ( (uint32_t) mval ) ) );
                }
                else if ( 1 == bits11_10 && 6 == bits23_21 && 2 == bits15_12 ) // lsrv
                {
                    uint64_t shift = mval;
                    if ( xregs )
                        shift = shift % 64;
                    else
                    {
                        nval = (uint32_t) nval;
                        shift = ( (uint32_t) shift ) & 0x1f;
                    }
                    regs[ d ] = ( nval >> shift );
                }
                else if ( 0 == bits11_10 && 6 == bits23_21 && 2 == bits15_12 ) // lslv
                {
                    uint64_t shift = mval;
                    if ( xregs )
                    {
                        shift = shift % 64;
                        regs[ d ] = ( nval << shift );
                    }
                    else
                    {
                        shift = ( (uint32_t) shift ) & 0x1f;
                        regs[ d ] = (uint32_t) ( nval << shift );
                    }
                }
                else if ( 0 == bits11_10 && 0 == bits23_21 && 0 == bits15_12 && 0 == bits11_10 ) // addc
                {
                    if ( xregs )
                        regs[ d ] = add_with_carry64( nval, mval, fC, false );
                    else
                        regs[ d ] = add_with_carry32( (uint32_t) nval, (uint32_t) mval, fC, false );
                }
                else if ( 3 == bits11_10 && 6 == bits23_21 && 2 == bits15_12 ) // RORV <Xd>, <Xn>, <Xm>
                {
                    if ( xregs )
                        regs[ d ] = shift_reg64( n, 3, mval );
                    else
                        regs[ d ] = shift_reg32( n, 3, mval );
                }
                else
                    unhandled();

                if ( !xregs )
                    regs[ d ] = (uint32_t) regs[ d ];
                break;
            }
            case 0x54: // b.cond
            {
                if ( check_conditional( opbits( 0, 4 ) ) )
                {
                    int64_t imm19 = opbits( 5, 19 );
                    imm19 <<= 2;
                    imm19 = sign_extend( imm19, 20 );
                    pc += imm19;
                    continue;
                }
                break;
            }
            case 0x18: // ldr wt, (literal)
            case 0x58: // ldr xt, (literal)
            {
                uint64_t imm19 = opbits( 5, 19 );
                uint64_t t = opbits( 0, 5 );
                bool xregs = ( 0 != opbit( 30 ) );
                uint64_t address = pc + ( imm19 << 2 );
                if ( 31 == t )
                    break;
                if ( xregs )
                    regs[ t ] = getui64( address );
                else
                    regs[ t ] = getui32( address );
                break;
            }
            case 0x3a: // CCMN <Wn>, #<imm>, #<nzcv>, <cond>  ;    CCMN <Wn>, <Wm>, #<nzcv>, <cond>       ;    ADCS <Wd>, <Wn>, <Wm>
            case 0xba: // CCMN <Wn>, <Wm>, #<nzcv>, <cond>    ;    CCMN <Xn>, <Xm>, #<nzcv>, <cond>       ;    ADCS <Xd>, <Xn>, <Xm>
            case 0x7a: // CCMP <Wn>, <Wm>, #<nzcv>, <cond>    ;    CCMP <Wn>, #<imm>, #<nzcv>, <cond>     ;    SBCS <Wd>, <Wn>, <Wm>
            case 0xfa: // CCMP <Xn>, <Xm>, #<nzcv>, <cond>    ;    CCMP <Xn>, #<imm>, #<nzcv>, <cond>     ;    SBCS <Xd>, <Xn>, <Xm>
            {
                uint64_t bits23_21 = opbits( 21, 3 );
                uint64_t bits15_10 = opbits( 10, 6 );
                uint64_t n = opbits( 5, 5 );
                bool xregs = ( 0 != ( 0x80 & hi8 ) );

                if ( 2 == bits23_21 )
                {
                    uint64_t o3 = opbit( 4 );
                    if ( 0 != o3 )
                        unhandled();

                    uint64_t cond = opbits( 12, 4 );
                    uint64_t nzcv = opbits( 0, 4 );
                    uint64_t o2 = opbits( 10, 2 );
                    if ( check_conditional( cond ) )
                    {
                        uint64_t op2 = 0;
                        if ( 0 == o2 ) // register
                        {
                            uint64_t m = opbits( 16, 5 );
                            op2 = val_reg_or_zr( m );
                        }
                        else if ( 2 == o2 ) // immediate
                            op2 = opbits( 16, 5 );
                        else
                            unhandled();

                        if ( 0 == ( hi8 & 0x40 ) ) // ccmn negative
                        {
                            if ( xregs )
                                op2 = - (int64_t) op2;
                            else
                                op2 = (uint32_t) ( - (int32_t) (uint32_t) op2 );
                        }

                        uint64_t op1 = val_reg_or_zr( n );
                        if ( xregs )
                            sub64( op1, op2, true );
                        else
                            sub32( (uint32_t) op1, (uint32_t) op2, true );
                    }
                    else
                        set_flags_from_nzcv( nzcv );
                }
                else if ( ( 0xfa == hi8 || 0x7a == hi8 ) && 0 == bits23_21 && 0 == bits15_10 ) // SBCS <Xd>, <Xn>, <Xm>
                {
                    uint64_t d = opbits( 0, 5 );
                    uint64_t m = opbits( 16, 5 );
                    uint64_t nval = val_reg_or_zr( n );
                    uint64_t mval = val_reg_or_zr( m );

                    uint64_t result = 0;
                    if ( xregs )
                        result = add_with_carry64( nval, ~mval, fC, true );
                    else
                        result = add_with_carry32( (uint32_t) nval, (uint32_t) ( ~ mval ), fC, true );
                    if ( 31 != d )
                        regs[ d ] = result;
                }
                else if ( ( 0x3a == hi8 || 0xba == hi8 ) && 0 == bits23_21 && 0 == bits15_10 ) // ADCS <Xd>, <Xn>, <Xm>
                {
                    uint64_t d = opbits( 0, 5 );
                    uint64_t m = opbits( 16, 5 );
                    uint64_t nval = val_reg_or_zr( n );
                    uint64_t mval = val_reg_or_zr( m );

                    uint64_t result = 0;
                    if ( xregs )
                        result = add_with_carry64( nval, mval, fC, true );
                    else
                        result = add_with_carry32( (uint32_t) nval, (uint32_t) mval, fC, true );
                    if ( 31 != d )
                        regs[ d ] = result;
                }
                else
                    unhandled();
                break;
            }
            case 0x71: // SUBS <Wd>, <Wn|WSP>, #<imm>{, <shift>}   ;   CMP <Wn|WSP>, #<imm>{, <shift>}
            case 0xf1: // SUBS <Xd>, <Xn|SP>, #<imm>{, <shift>}    ;   cmp <xn|SP>, #imm [,<shift>]
            case 0x31: // ADDS <Wd>, <Wn|WSP>, #<imm>{, <shift>}  ;    CMN <Wn|WSP>, #<imm>{, <shift>}
            case 0xb1: // ADDS <Xd>, <Xn|SP>, #<imm>{, <shift>}   ;    CMN <Xn|SP>, #<imm>{, <shift>}
            {
                uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
                uint64_t imm12 = opbits( 10, 12 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                bool is_sub = ( 0 != ( 0x40 & hi8 ) );
                bool shift12 = opbit( 22 );
                if ( shift12 )
                    imm12 <<= 12;

                uint64_t result;

                if ( xregs )
                {
                    if ( is_sub )
                        result = sub64( regs[ n ], imm12, true );
                    else
                        result = add_with_carry64( regs[ n ], imm12, false, true );
                }
                else
                {
                    if ( is_sub )
                        result = sub32( (uint32_t) regs[ n ], (uint32_t) imm12, true );
                    else
                        result = add_with_carry32( (uint32_t) regs[ n ], (uint32_t) imm12, false, true );
                }

                if ( 31 != d )
                    regs[ d ] = result;
                break;
            }
            case 0x0b: // ADD <Wd|WSP>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}      ;    ADD <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
            case 0x2b: // ADDS <Wd>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}         ;    ADDS <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
            case 0x4b: // SUB <Wd|WSP>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}      ;    SUB <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
            case 0x6b: // SUBS <Wd>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}         ;    SUBS <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
            case 0x8b: // ADD <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}      ;    ADD <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
            case 0xab: // ADDS <Xd>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}        ;    ADDS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
            case 0xcb: // SUB <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}      ;    SUB <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
            case 0xeb: // SUBS <Xd>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}        ;    SUBS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
            {
                uint64_t extended = opbit( 21 );
                uint64_t issub = ( 0 != ( 0x40 & hi8 ) );
                uint64_t setflags = ( 0 != ( 0x20 & hi8 ) );
                uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
                uint64_t m = opbits( 16, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t offset = 0;
                uint64_t nvalue = regs[ n ];

                if ( 1 == extended ) // ADD <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
                {
                    uint64_t option = opbits( 13, 3 );
                    uint64_t imm3 = opbits( 10, 3 );
                    offset = extend_reg( m, option, imm3 );
                }
                else // ADD <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
                {
                    uint64_t shift = opbits( 22, 2 );
                    uint64_t imm6 = opbits( 10, 6 );
                    if ( xregs )
                        offset = shift_reg64( m, shift, imm6 );
                    else
                        offset = shift_reg32( m, shift, imm6 );
                    if ( 31 == n )
                        nvalue = 0;
                }

                uint64_t result = 0;
                if ( issub )
                {
                    if ( xregs )
                        result = sub64( nvalue, offset, setflags );
                    else
                        result = sub32( (uint32_t) nvalue, (uint32_t) offset, setflags );
                }
                else
                {
                    if ( xregs )
                        result = add_with_carry64( nvalue, offset, false, setflags );
                    else
                        result = add_with_carry32( (uint32_t) nvalue, (uint32_t) offset, false, setflags );
                }

                if ( ( !setflags ) || ( 31 != d ) )
                    regs[ d ] = result;
                break;
            }
            case 0x94: case 0x95: case 0x96: case 0x97: // bl offset. The lower 2 bits of this are the high part of the offset
            {
                int64_t offset = ( opbits( 0, 26 ) << 2 );
                offset = sign_extend( offset, 27 );
                regs[ 30 ] = pc + 4;
                pc += offset;
                //trace_vregs();
                continue;
            }
            case 0x11: // add <wd|SP>, <wn|SP>, #imm [,<shift>]
            case 0x51: // sub <wd|SP>, <wn|SP>, #imm [,<shift>]
            case 0x91: // add <xd|SP>, <xn|SP>, #imm [,<shift>]
            case 0xd1: // sub <xd|SP>, <xn|SP>, #imm [,<shift>]
            {
                bool sf = ( 0 != opbit( 31 ) );
                bool sh = ( 0 != opbit( 22 ) );
                uint64_t imm12 = opbits( 10, 12 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t op1 = regs[ n ];
                uint64_t op2 = sh ? ( imm12 << 12 ) : imm12;
                uint64_t result;
                if ( hi8 & 0x40 ) // sub
                {
                    if ( sf )
                        result = sub64( op1, op2, false );
                    else
                        result = sub32( (uint32_t) op1, (uint32_t) op2, false );
                }
                else
                {
                    if ( sf )
                        result = add_with_carry64( op1, op2, false, false );
                    else
                        result = add_with_carry32( (uint32_t) op1, (uint32_t) op2, false, false );
                }
                regs[ d ] = result;
                break;
            }
            case 0x28: // ldp/stp 32 post index                   STP <Wt1>, <Wt2>, [<Xn|SP>], #<imm>     ;    LDP <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
            case 0xa8: // ldp/stp 64 post-index                   STP <Xt1>, <Xt2>, [<Xn|SP>], #<imm>     ;    LDP <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
            case 0x29: // ldp/stp 32 pre-index and signed offset: STP <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!    ;    STP <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
                       //                                         LDP <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!    ;    LDP <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
            case 0xa9: // ldp/stp 64 pre-index and signed offset: STP <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!    ;    STP <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
                       //                                         LDP <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!    ;    LDP <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
            case 0x68: // ldp 32-bit sign extended                LDPSW <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
            case 0x69: // ldp 32-bit sign extended                LDPSW <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!  ;    LDPSW <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
            {
                bool xregs = ( 0 != opbit( 31 ) );
                uint64_t t1 = opbits( 0, 5 );
                uint64_t t2 = opbits( 10, 5 );
                uint64_t n = opbits( 5, 5 );
                int64_t imm7 = sign_extend( opbits( 15, 7 ), 6 ) << ( xregs ? 3 : 2 );
                uint64_t variant = opbits( 23, 2 );
                if ( 0 == variant )
                    unhandled();

                bool postIndex = ( 1 == variant );
                bool preIndex = ( 3 == variant );
                bool signedOffset = ( 2 == variant );
                uint64_t address = regs[ n ];
                if ( preIndex )
                    address += imm7;
                uint64_t effectiveAddress = address + ( signedOffset ? imm7 : 0 );

                if ( 0 == opbit( 22 ) ) // bit 22 is 0 for stp
                {
                    uint64_t t1val = val_reg_or_zr( t1 );
                    uint64_t t2val = val_reg_or_zr( t2 );

                    if ( xregs )
                    {
                        setui64( effectiveAddress, t1val );
                        setui64( effectiveAddress + 8, t2val );
                    }
                    else
                    {
                        setui32( effectiveAddress, (uint32_t) t1val );
                        setui32( effectiveAddress + 4, (uint32_t) t2val );
                    }
                }
                else // 1 means ldp
                {
                    // LDP <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
                    // LDP <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
                    // LDP <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]

                    if ( xregs )
                    {
                        if ( 31 != t1 )
                            regs[ t1 ] = getui64( effectiveAddress );
                        if ( 31 != t2 )
                            regs[ t2 ] = getui64( effectiveAddress + 8 );
                    }
                    else
                    {
                        bool se = ( 0 != ( hi8 & 0x40 ) );
                        if ( 31 != t1 )
                        {
                            regs[ t1 ] = getui32( effectiveAddress );
                            if ( se )
                                regs[ t1 ] = sign_extend( regs[ t1 ], 31 );
                        }
                        if ( 31 != t2 )
                        {
                            regs[ t2 ] = getui32( effectiveAddress + 4 );
                            if ( se )
                                regs[ t2 ] = sign_extend( regs[ t2 ], 31 );
                        }
                    }
                }

                if ( postIndex )
                    address += imm7;

                if ( preIndex || postIndex )
                    regs[ n ] = address;
                break;
            }
            case 0x32: // ORR <Wd|WSP>, <Wn>, #<imm>
            case 0xb2: // ORR <Xd|SP>, <Xn>, #<imm>
            {
                uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
                uint64_t N_immr_imms = opbits( 10, 13 );
                uint64_t op2 = decode_logical_immediate( N_immr_imms, xregs ? 64 : 32 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t nvalue = val_reg_or_zr( n );

                regs[ d ] = nvalue | op2;
                if ( !xregs )
                    regs[ d ] = (uint32_t) regs[ d ];
                break;
            }
            case 0x4a: // EOR <Wd>, <Wn>, <Wm>{, <shift> #<amount>}    ;    EON <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
            case 0xca: // EOR <Xd>, <Xn>, <Xm>{, <shift> #<amount>}    ;    EON <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
            case 0x2a: // ORR <Wd>, <Wn>, <Wm>{, <shift> #<amount>}    ;    ORN <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
            case 0xaa: // ORR <Xd>, <Xn>, <Xm>{, <shift> #<amount>}    ;    ORN <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
            {
                uint64_t shift = opbits( 22, 2 );
                uint64_t N = opbit( 21 );
                uint64_t m = opbits( 16, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t imm6 = opbits( 10, 6 );
                uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
                bool eor = ( 2 == opbits( 29, 2 ) );

                if ( 31 == d )
                    break;

                uint64_t nval = val_reg_or_zr( n );
                if ( ( 0 == imm6 ) && ( 31 == n ) && ( 0 == shift ) && ( 0 == N ) ) // mov
                    regs[ d ] = val_reg_or_zr( m );
                else if ( ( 0 == shift ) && ( 0 == imm6 ) )
                {
                    uint64_t mval = val_reg_or_zr( m );
                    if ( eor )
                        regs[ d ] = nval ^ ( ( 0 == N ) ? mval : ~mval );
                    else
                        regs[ d ] = nval | ( ( 0 == N ) ? mval : ~mval );
                }
                else
                {
                    uint64_t mval = xregs ? shift_reg64( m, shift, imm6 ) : shift_reg32( m, shift, imm6 );
                    if ( eor )
                        regs[ d ] = nval ^ ( ( 0 == N ) ? mval : ~mval );
                    else
                        regs[ d ] = nval | ( ( 0 == N ) ? mval : ~mval );
                }

                if ( !xregs )
                    regs[ d ] = (uint32_t) regs[ d ];
                break;
            }
            case 0x33: // BFM <Wd>, <Wn>, #<immr>, #<imms>       // original bits intact
            case 0xb3: // BFM <Xd>, <Xn>, #<immr>, #<imms>
            case 0x13: // SBFM <Wd>, <Wn>, #<immr>, #<imms>    ;    EXTR <Wd>, <Wn>, <Wm>, #<lsb>
            case 0x93: // SBFM <Xd>, <Xn>, #<immr>, #<imms>    ;    EXTR <Xd>, <Xn>, <Xm>, #<lsb>
            case 0x53: // UBFM <Wd>, <Wn>, #<immr>, #<imms>      // unmodified bits set to 0
            case 0xd3: // UBFM <Xd>, <Xn>, #<immr>, #<imms>
            {
                uint64_t imms = opbits( 10, 6 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t bit23 = opbit( 23 );
                bool xregs = ( 0 != opbit( 31 ) );

                if ( 31 == d )
                    break;

                if ( bit23 && ( 0x13 == ( 0x7f & hi8 ) ) ) // EXTR. rotate right preserving bits shifted out in the high bits
                {
                    uint64_t m = opbits( 16, 5 );

                    if ( xregs )
                    {
                        uint64_t nval = val_reg_or_zr( n );
                        uint64_t mval = val_reg_or_zr( m );
                        regs[ d ] = ( mval >> imms ) | ( nval << ( 64 - imms ) );
                    }
                    else
                    {
                        uint32_t nval = (uint32_t) val_reg_or_zr( n );
                        uint32_t mval = (uint32_t) val_reg_or_zr( m );
                        regs[ d ] = ( mval >> imms ) | ( nval << ( 32 - imms ) );
                    }
                }
                else // others
                {
                    uint64_t immr = opbits( 16, 6 );
                    uint64_t src = val_reg_or_zr( n );
                    uint64_t result = ( 0x33 == hi8 || 0xb3 == hi8 ) ? regs[ d ] : 0;   // restore original bits for BFM
                    uint64_t dpos = 0;

                    if ( imms >= immr )
                    {
                        uint64_t len = imms - immr + 1;
                        result &= ( ~ one_bits( len ) );
                        result |= get_bits( src, immr, len );
                        dpos = len;
                    }
                    else
                    {
                        uint64_t len = imms + 1;
                        uint64_t reg_size = xregs ? 64 : 32;
                        dpos = reg_size - immr;
                        uint64_t tmp = get_bits( src, 0, len );
                        result = plaster_bits( result, tmp, dpos, len );
                        dpos += len;
                    }

                    assert( 0 != dpos );
                    if ( ( 0x13 == hi8 || 0x93 == hi8 ) && get_bit( result, dpos - 1 ) ) // SBFM
                    {
                        //tracer.Trace( "  dpos %llu, most significant bit set, sbfm, extending %llx\n", dpos, result );
                        result = sign_extend( result, dpos - 1 );
                    }

                    if ( 0 == ( hi8 & 0x80 ) )
                        result = (uint32_t) result;
                    regs[ d ] = result;
                    //tracer.Trace( "  source %#llx changed to result %#llx\n", s, result );
                }
                break;
            }
            case 0x0a: // AND <Wd>, <Wn>, <Wm>{, <shift> #<amount>}     ;    BIC <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
            case 0x6a: // ANDS <Wd>, <Wn>, <Wm>{, <shift> #<amount>}    ;    BICS <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
            case 0x8a: // AND <Xd>, <Xn>, <Xm>{, <shift> #<amount>}     ;    BIC <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
            case 0xea: // ANDS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}    ;    BICS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
            {
                uint64_t shift = opbits( 22, 2 );
                uint64_t N = opbit( 21 ); // BICS -- complement
                uint64_t m = opbits( 16, 5 );
                uint64_t imm6 = opbits( 10, 6 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                bool set_flags = ( 0x60 == ( hi8 & 0x60 ) );
                bool xregs = ( 0 != ( hi8 & 0x80 ) );

                uint64_t op2;
                if ( xregs )
                {
                    op2 = shift_reg64( m, shift, imm6 );
                    if ( N )
                        op2 = ~op2;
                }
                else
                {
                    op2 = shift_reg32( m, shift, imm6 );
                    if ( N )
                        op2 = (uint32_t) ( ~op2 );
                }

                uint64_t result = ( regs[ n ] & op2 );

                if ( set_flags )
                {
                    fZ = ( 0 == result );
                    fV = fC = false;
                    fN = xregs ? get_bit( result, 63 ) : get_bit( result, 31 );
                }

                if ( 31 != d )
                    regs[ d ] = result;
                break;
            }
            case 0x10: case 0x30: case 0x50: case 0x70: // ADR <Xd>, <label>
            case 0x90: case 0xb0: case 0xd0: case 0xf0: // ADRP <Xd>, <label>
            {
                uint64_t d = opbits( 0, 5 );
                if ( 31 == d )
                    break;
                int64_t imm = ( ( op >> 3 ) & 0x1ffffc );  // 19 bits with bottom two bits 0
                imm |= opbits( 29, 2 );  // two low bits
                imm = sign_extend( imm, 20 );
                if ( get_bit( hi8, 7 ) ) // adrp
                {
                    imm <<= 12;
                    imm += ( pc & ( ~0xfff ) );
                }
                else // adr
                    imm += pc;
                regs[ d ] = imm;
                break;
            }
            case 0x52: // MOVZ <Wd>, #<imm>{, LSL #<shift>}    ;    EOR <Wd|WSP>, <Wn>, #<imm>
            case 0xd2: // MOVZ <Xd>, #<imm>{, LSL #<shift>}    ;    EOR <Xd|SP>, <Xn>, #<imm>
            {
                bool xregs = ( 0 != ( hi8 & 0x80 ) );
                uint64_t d = opbits( 0, 5 );
                uint64_t bit23 = opbit( 23 );

                if ( bit23 ) // movz xd, imm16
                {
                    if ( 31 == d )
                        break;
                    uint64_t imm16 = opbits( 5, 16 );
                    uint64_t hw = opbits( 21, 2 );
                    regs[ d ] = ( imm16 << ( hw * 16 ) );
                }
                else // EOR
                {
                    uint64_t N_immr_imms = opbits( 10, 13 );
                    uint64_t op2 = decode_logical_immediate( N_immr_imms, xregs ? 64 : 32 );
                    uint64_t n = opbits( 5, 5 );
                    uint64_t nvalue = val_reg_or_zr( n );
                    regs[ d ] = nvalue ^ op2;
                    if ( !xregs )
                        regs[ d ] = (uint32_t) regs[ d ];
                }
                break;
            }
            case 0x36: // TBZ <R><t>, #<imm>, <label>
            case 0x37: // TBNZ <R><t>, #<imm>, <label>
            case 0xb6: // TBZ <R><t>, #<imm>, <label> where high bit is prepended to b40 bit selector for 6 bits total
            case 0xb7: // TBNZ <R><t>, #<imm>, <label> where high bit is prepended to b40 bit selector for 6 bits total
            {
                uint64_t b40 = opbits( 19, 5 );
                if ( 0 != ( 0x80 & hi8 ) )
                    b40 |= 0x20;
                uint64_t t = opbits( 0, 5 );
                uint64_t mask = ( 1ull << b40 );
                bool isset = ( 0 != ( regs[ t ] & mask ) );
                bool truecheck = ( hi8 & 1 );
                if ( isset == truecheck )
                {
                    int64_t imm14 = (int64_t) sign_extend( ( opbits( 5, 14 ) << 2 ), 15 );
                    pc += imm14;
                    continue;
                }
                break;
            }
            case 0x12: // MOVN <Wd>, #<imm>{, LSL #<shift>}   ;    AND <Wd|WSP>, <Wn>, #<imm>
            case 0x92: // MOVN <Xd>, #<imm16>, LSL #<shift>   ;    AND <Xd|SP>, <Xn>, #<imm>    ;    MOV <Xd>, #<imm>
            {
                uint64_t d = opbits( 0, 5 );
                uint64_t bit23 = opbit( 23 );
                bool xregs = ( 0 != ( hi8 & 0x80 ) );
                if ( bit23 ) // MOVN
                {
                    if ( 31 == d )
                        break;
                    uint64_t imm16 = opbits( 5, 16 );
                    uint64_t hw = opbits( 21, 2 );
                    hw *= 16;
                    imm16 <<= hw;
                    imm16 = ~imm16;

                    if ( 0x12 == hi8 )
                    {
                        if ( hw > 16 )
                            unhandled();
                        imm16 = (uint32_t) imm16;
                    }
                    regs[ d ] = imm16;
                }
                else // AND
                {
                    uint64_t N_immr_imms = opbits( 10, 13 );
                    uint64_t op2 = decode_logical_immediate( N_immr_imms, xregs ? 64 : 32 );
                    uint64_t n = opbits( 5, 5 );
                    uint64_t nval = val_reg_or_zr( n );
                    regs[ d ] = ( nval & op2 );
                    if ( !xregs )
                        regs[ d ] = (uint32_t) regs[ d ];
                }
                break;
            }
            case 0x34: // CBZ <Wt>, <label>
            case 0x35: // CBNZ <Wt>, <label>
            case 0xb4: // CBZ <Xt>, <label>
            case 0xb5: // CBNZ <Xt>, <label>
            {
                uint64_t t = opbits( 0, 5 );
                uint64_t val = val_reg_or_zr( t );
                bool zero_check = ( 0 == ( hi8 & 1 ) );
                if ( 0 == ( 0x80 & hi8 ) )
                    val = (uint32_t) val;

                if ( zero_check == ( 0 == val ) )
                {
                    int64_t imm19 = ( ( op >> 3 ) & 0x1ffffc ); // two low bits are 0
                    imm19 = sign_extend( imm19, 20 );
                    pc += imm19;
                    continue;
                }
                break;
            }
            case 0xd4: // SVC
            {
                uint64_t bit23 = opbit( 23 );
                uint64_t hw = opbits( 21, 2 );

                if ( !bit23 && ( 0 == hw ) )
                {
                    uint64_t op2 = opbits( 2, 3 );
                    uint64_t ll = opbits( 0, 2 );
                    if ( ( 0 == op2 ) && ( 1 == ll ) ) // svc imm16 supervisor call
                        emulator_invoke_svc( *this );
                    else
                        unhandled();
                }
                else
                    unhandled();
                break;
            }
            case 0xd5: // MSR / MRS
            {
                uint64_t bits2322 = opbits( 22, 2 );
                if ( 0 != bits2322 )
                    unhandled();

                if ( 0xd503201f == op ) // nop
                    break;

                uint64_t upper20 = opbits( 12, 20 );
                uint64_t lower8 = opbits( 0, 8 );
                if ( ( 0xd5033 == upper20 ) && ( 0xbf == lower8 ) ) // dmb -- no memory barries are needed due to just one thread and core
                    break;

                uint64_t l = opbit( 21 );
                uint64_t op0 = opbits( 19, 2 );
                uint64_t op1 = opbits( 16, 3 );
                uint64_t op2 = opbits( 5, 3 );
                uint64_t n = opbits( 12, 4 );
                uint64_t m = opbits( 8, 4 );
                uint64_t t = opbits( 0, 5 );

                if ( l ) // MRS <Xt>, (<systemreg>|S<op0>_<op1>_<Cn>_<Cm>_<op2>).   read system register
                {
                    if ( ( 3 == op0 ) && ( 14 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 2 == op2 ) ) // cntvct_el0 counter-timer virtual count register
                    {
                        system_clock::duration d = system_clock::now().time_since_epoch();
                        regs[ t ] = duration_cast<nanoseconds>( d ).count();
                    }
                    else if ( ( 3 == op0 ) && ( 14 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 0 == op2 ) ) // cntfrq_el0 counter-timer frequency register
                        regs[ t ] = 1000000000; // nanoseconds = billionths of a second
                    else if ( ( 3 == op0 ) && ( 0 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 7 == op2 ) ) // DCZID_EL0. data cache block size for dc zva instruction
                        regs[ t ] = 4; // doesn't matter becasuse there is no caching in the emulator
                    else if ( ( 3 == op0 ) && ( 0 == n ) && ( 0 == op1 ) && ( 0 == m ) && ( 0 == op2 ) ) // mrs x, midr_el1
                        regs[ t ] = 0x595a5449; // ITZY don't you know you have a superpower? // my dev machine: 0x410fd4c0;
                    else if ( ( 3 == op0 ) && ( 13 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 2 == op2 ) ) // software thread id
                        regs[ t ] = tpidr_el0;
                    else if ( ( 3 == op0 ) && ( 4 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 0 == op2 ) ) // mrs x, fpcr
                        regs[ t ] = fpcr;
                    else if ( ( 3 == op0 ) && ( 4 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 1 == op2 ) ) // mrs x, fpsr
                        regs[ t ] = 0;
                    else
                        unhandled();
                }
                else // MSR.   write system register
                {
                    if ( ( 3 == op0 ) && ( 13 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 2 == op2 ) ) // software thread id
                        tpidr_el0 = regs[ t ];
                    else if ( ( 0 == op0 ) && ( 2 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 2 == op2 ) )
                    {
                        // branch target identification (ignore)
                    }
                    else if ( ( 1 == op0 ) && ( 7 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 1 == op2 ) )
                        memset( getmem( regs[ t ] ), 0, 4 * 32 ); // dc zva <Xt>
                    else if ( ( 0 == op0 ) && ( 2 == n ) && ( 3 == op1 ) && ( 0 == m ) && ( 7 == op2 ) ) // xpaclri
                    {
                        // do nothing
                    }
                    else if ( ( 3 == op0 ) && ( 4 == n ) && ( 3 == op1 ) && ( 4 == m ) && ( 0 == op2 ) ) // msr fpcr, xt
                    {
                        // If FPCR.AH (bit 1) is 1, then the following instructions use Round to Nearest mode regardless of the value of this bit:
                        //   The FRECPE, FRECPS, FRECPX, FRSQRTE, and FRSQRTS instructions.
                        //   The BFCVT, BFCVTN, BFCVTN2, BFCVTNT, BFMLALB, and BFMLALT instructions.
                        // RMode is in bits 23-22:
                        //   00 = round to nearest RN
                        //   01 = round towards plus infinity RP
                        //   10 = round towards minus infinity RM
                        //   11 = round towards zero RZ
                        fpcr = regs[ t ];
                    }
                    else
                        unhandled();
                }
                break;
            }
            case 0x2e: case 0x6e: // CMEQ <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    CMHS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMAXP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // BIT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>     ;    UMINP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    BIF <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // EOR <Vd>.<T>, <Vn>.<T>, <Vm>.<T>     ;    SUB <Vd>.<T>, <Vn>.<T>, <Vm>.<T>     ;    UMULL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                                  // MLS <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>] ;  BSL <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;    FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // EXT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<index> ;  INS <Vd>.<Ts>[<index1>], <Vn>.<Ts>[<index2>]  ;    UADDLV <V><d>, <Vn>.<T>
                                  // USHL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    FADDP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    FNEG <Vd>.<T>, <Vn>.<T>
                                  // CMHI <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UADDW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>  ;   FDIV <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // UMAXV <V><d>, <Vn>.<T> ; UMINV <V><d>, <Vn>.<T>    ;    UMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // FMINNMV S<d>, <Vn>.4S  ; FMAXNMV S<d>, <Vn>.4S     ;    MLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>     ;    UCVTF <Vd>.<T>, <Vn>.<T>
                                  // NEG <Vd>.<T>, <Vn>.<T> ; EXT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<index>            ;    FCVTZU <Vd>.<T>, <Vn>.<T>
                                  // UMLAL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>           ;    USUBW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
                                  // UADDL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>           ;    USUBL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                                  // FSQRT <Vd>.<T>, <Vn>.<T>             ;    UMLSL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb> ; NOT <Vd>.<T>, <Vn>.<T>
                                  // UADDLP <Vd>.<Ta>, <Vn>.<Tb>          ;    UADALP <Vd>.<Ta>, <Vn>.<Tb>          ;    CMLE <Vd>.<T>, <Vn>.<T>, #0
                                  // UQXTN{2} <Vd>.<Tb>, <Vn>.<Ta>        ;    FCMGT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    FCMGE <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // CMGE <Vd>.<T>, <Vn>.<T>, #0          ;    REV32 <Vd>.<T>, <Vn>.<T>
            {
                uint64_t Q = opbit( 30 );
                uint64_t m = opbits( 16, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t size = opbits( 22, 2 );
                uint64_t opc2 = opbits( 22, 2 );
                uint64_t bit23 = opbit( 23 );
                uint64_t bit21 = opbit( 21 );
                uint64_t bit15 = opbit( 15 );
                uint64_t bit10 = opbit( 10 );
                uint64_t opcode = opbits( 10, 6 );
                uint64_t esize = 8ull << size;
                uint64_t ebytes = esize / 8;
                uint64_t datasize = 64ull << Q;
                uint64_t elements = datasize / esize;
                uint64_t bits23_21 = opbits( 21, 3 );
                uint64_t opcode7 = opbits( 10, 7 );
                uint64_t bits20_17 = opbits( 17, 4 );
                uint64_t bits20_16 = opbits( 16, 5 );
                uint64_t bits16_10 = opbits( 10, 7 );
                uint64_t bits15_10 = opbits( 10, 6 );
                //tracer.Trace( "elements: %llu, size %llu, esize %llu, datasize %llu, ebytes %llu, opcode %llu\n", elements, size, esize, datasize, ebytes, opcode );

                if ( bit21 )
                {
                    if ( 0 == bits20_16 && 2 == bits15_10 ) // REV32 <Vd>.<T>, <Vn>.<T>
                    {
                        if ( size > 1 )
                            unhandled();
                        uint64_t csize = 32;
                        uint64_t containers = datasize / csize;
                        vec16_t target;
                        for ( uint64_t c = 0; c < containers; c++ )
                            target.set32( c, flip_endian32( vregs[ n ].get32( c ) ) );
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_17 && 0x22 == bits16_10 ) // CMGE <Vd>.<T>, <Vn>.<T>, #0
                    {
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set8( e, ( (int8_t) vregs[ n ].get8( e ) >= 0 ) ? ~0 : 0 );
                            else if ( 2 == ebytes )
                                target.set16( e, ( (int16_t) vregs[ n ].get16( e ) >= 0 ) ? ~0 : 0 );
                            else if ( 4 == ebytes )
                                target.set32( e, ( (int32_t) vregs[ n ].get32( e ) >= 0 ) ? ~0 : 0 );
                            else if ( 8 == ebytes )
                                target.set64( e, ( (int64_t) vregs[ n ].get64( e ) >= 0 ) ? ~0 : 0 );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 1 == bits23_21 && 0 == bits20_16 && 0x16 == bits15_10 ) // NOT <Vd>.<T>, <Vn>.<T>. AKA MVN
                    {
                        vregs[ d ].set64( 0, ~ vregs[ n ].get64( 0 ) );
                        if ( Q )
                            vregs[ d ].set64( 1, ~ vregs[ n ].get64( 1 ) );
                    }
                    else if ( !bit23 && 0x39 == bits15_10 ) // FCMGE <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        esize = 32 << sz;
                        ebytes = esize / 8;
                        elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                target.set32( e, ( vregs[ n ].getf( e ) >= vregs[ m ].getf( e ) ) ? ~0 : 0 );
                            else if ( 8 == ebytes )
                                target.set64( e, ( vregs[ n ].getd( e ) >= vregs[ m ].getd( e ) ) ? ~0 : 0 );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( bit23 && 0x39 == bits15_10 ) // FCMGT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        esize = 32 << sz;
                        ebytes = esize / 8;
                        elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                target.set32( e, ( vregs[ n ].getf( e ) > vregs[ m ].getf( e ) ) ? ~0 : 0 );
                            else if ( 8 == ebytes )
                                target.set64( e, ( vregs[ n ].getd( e ) > vregs[ m ].getd( e ) ) ? ~0 : 0 );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 1 == bits20_16 && 0x12 == bits15_10 ) // UQXTN{2} <Vd>.<Tb>, <Vn>.<Ta>
                    {
                        datasize = 64;
                        elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                            {
                                uint16_t ui16 = vregs[ n ].get16( e );
                                uint8_t ui8 = ( ui16 > UINT8_MAX ) ? UINT8_MAX : (uint8_t) ui16;
                                if ( Q )
                                    target.set8( 8 + e, ui8 );
                                else
                                {
                                    target.set8( e, ui8 );
                                    target.set8( 8 + e, 0 );
                                }
                            }
                            else if ( 2 == ebytes )
                            {
                                uint32_t ui32 = vregs[ n ].get32( e );
                                uint16_t ui16 = ( ui32 > UINT16_MAX ) ? UINT16_MAX : (uint16_t) ui32;
                                if ( Q )
                                    target.set16( 4 + e, ui16 );
                                else
                                {
                                    target.set16( e, ui16 );
                                    target.set16( 4 + e, 0 );
                                }
                            }
                            else if ( 4 == ebytes )
                            {
                                uint64_t ui64 = vregs[ n ].get64( e );
                                uint32_t ui32 = ( ui64 > UINT32_MAX ) ? UINT32_MAX : (uint32_t) ui64;
                                if ( Q )
                                    target.set32( 2 + e, ui32 );
                                else
                                {
                                    target.set32( e, ui32 );
                                    target.set32( 2 + e, 0 );
                                }
                            }
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_17 && 0x26 == bits16_10 ) // CMLE <Vd>.<T>, <Vn>.<T>, #0
                    {
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set8( e, ( (int8_t) vregs[ n ].get8( e ) <= 0 ) ? ~0 : 0 );
                            else if ( 2 == ebytes )
                                target.set16( e, ( (int16_t) vregs[ n ].get16( e ) <= 0 ) ? ~0 : 0 );
                            else if ( 4 == ebytes )
                                target.set32( e, ( (int32_t) vregs[ n ].get32( e ) <= 0 ) ? ~0 : 0 );
                            else if ( 8 == ebytes )
                                target.set64( e, ( (int64_t) vregs[ n ].get64( e ) <= 0 ) ? ~0 : 0 );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_17 && 0x1a == bits16_10 ) // UADALP <Vd>.<Ta>, <Vn>.<Tb>
                    {
                        elements = datasize / ( 2 * esize );
                        vec16_t target = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, target.get16( e ) + vregs[ n ].get8( 2 * e ) + vregs[ n ].get8( 2 * e + 1 ) );
                            else if ( 2 == ebytes )
                                target.set32( e, target.get32( e ) + vregs[ n ].get16( 2 * e ) + vregs[ n ].get16( 2 * e + 1 ) );
                            else if ( 4 == ebytes )
                                target.set64( e, target.get64( e ) + vregs[ n ].get32( 2 * e ) + vregs[ n ].get32( 2 * e + 1 ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_17 && 0xa == bits16_10 ) // UADDLP <Vd>.<Ta>, <Vn>.<Tb>
                    {
                        elements = datasize / ( 2 * esize );
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, vregs[ n ].get8( 2 * e ) + vregs[ n ].get8( 2 * e + 1 ) );
                            else if ( 2 == ebytes )
                                target.set32( e, vregs[ n ].get16( 2 * e ) + vregs[ n ].get16( 2 * e + 1 ) );
                            else if ( 4 == ebytes )
                                target.set64( e, vregs[ n ].get32( 2 * e ) + vregs[ n ].get32( 2 * e + 1 ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( bit23 && 0 == bits20_17 && 0x7e == bits16_10 ) // FSQRT <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        esize = 32ull << sz;
                        ebytes = esize / 8;
                        elements = datasize / esize;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                vregs[ d ].setf( e, sqrtf( vregs[ n ].getf( e ) ) );
                            else if ( 8 == ebytes )
                                vregs[ d ].setd( e, sqrt( vregs[ n ].getd( e ) ) );
                            else
                                unhandled();
                        }
                    }
                    else if ( 8 == bits15_10 ) // USUBL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                    {
                        datasize = 64;
                        elements = datasize / esize;
                        vec16_t target = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, ( (uint16_t) vregs[ n ].get8( e + ( Q ? 8 : 0 ) ) - (uint16_t) vregs[ m ].get8( e + ( Q ? 8 : 0 ) ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, ( (uint32_t) vregs[ n ].get16( e + ( Q ? 4 : 0 ) ) - (uint32_t) vregs[ m ].get16( e + ( Q ? 4 : 0 ) ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, ( (uint64_t) vregs[ n ].get32( e + ( Q ? 2 : 0 ) ) - (uint64_t) vregs[ m ].get32( e + ( Q ? 2 : 0 ) ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits15_10 ) // UADDL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                    {
                        datasize = 64;
                        elements = datasize / esize;
                        vec16_t target = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, ( (uint16_t) vregs[ n ].get8( e + ( Q ? 8 : 0 ) ) + (uint16_t) vregs[ m ].get8( e + ( Q ? 8 : 0 ) ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, ( (uint32_t) vregs[ n ].get16( e + ( Q ? 4 : 0 ) ) + (uint32_t) vregs[ m ].get16( e + ( Q ? 4 : 0 ) ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, ( (uint64_t) vregs[ n ].get32( e + ( Q ? 2 : 0 ) ) + (uint64_t) vregs[ m ].get32( e + ( Q ? 2 : 0 ) ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0xc == bits15_10 ) // USUBW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>. Note: the official docs confuse the first and second operands wr
                    {
                        datasize = 64;
                        elements = datasize / esize;
                        vec16_t target = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, vregs[ n ].get16( e ) - (uint16_t) vregs[ m ].get8( e + ( Q ? 8 : 0 ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, vregs[ n ].get32( e ) - (uint32_t) vregs[ m ].get16( e + ( Q ? 4 : 0 ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, vregs[ n ].get64( e ) - (uint64_t) vregs[ m ].get32( e + ( Q ? 2 : 0 ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0x20 == bits15_10 ) // UMLAL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                    {
                        datasize = 64;
                        elements = datasize / esize;
                        vec16_t target = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, target.get16( e ) + ( (uint16_t) vregs[ n ].get8( e + ( Q ? 8 : 0 ) ) * (uint16_t) vregs[ m ].get8( e + ( Q ? 8 : 0 ) ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, target.get32( e ) + ( (uint32_t) vregs[ n ].get16( e + ( Q ? 4 : 0 ) ) * (uint32_t) vregs[ m ].get16( e + ( Q ? 4 : 0 ) ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, target.get64( e ) + ( (uint64_t) vregs[ n ].get32( e + ( Q ? 2 : 0 ) ) * (uint64_t) vregs[ m ].get32( e + ( Q ? 2 : 0 ) ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0x28 == bits15_10 ) // UMLSL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                    {
                        datasize = 64;
                        elements = datasize / esize;
                        vec16_t target = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, target.get16( e ) - ( (uint16_t) vregs[ n ].get8( e + ( Q ? 8 : 0 ) ) * (uint16_t) vregs[ m ].get8( e + ( Q ? 8 : 0 ) ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, target.get32( e ) - ( (uint32_t) vregs[ n ].get16( e + ( Q ? 4 : 0 ) ) * (uint32_t) vregs[ m ].get16( e + ( Q ? 4 : 0 ) ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, target.get64( e ) - ( (uint64_t) vregs[ n ].get32( e + ( Q ? 2 : 0 ) ) * (uint64_t) vregs[ m ].get32( e + ( Q ? 2 : 0 ) ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( !bit23 && 0 == bits20_17 && 0x76 == bits16_10 ) // UCVTF <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        esize = 32ull << sz;
                        ebytes = esize / 8;
                        elements = datasize / esize;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                vregs[ d ].setf( e, (float) vregs[ n ].get32( e ) );
                            else if ( 8 == ebytes )
                                vregs[ d ].setd( e, (double) vregs[ n ].get64( e ) );
                        }
                    }
                    else if ( bit23 && 0 == bits20_17 && 0x6e == bits16_10 ) // FCVTZU <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        esize = 32ull << sz;
                        ebytes = esize / 8;
                        elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                target.set32( e, double_to_fixed_uint32( (double) vregs[ n ].getf( e ), 0, FPRounding_ZERO ) );
                            else if ( 8 == ebytes )
                                target.set64( e, double_to_fixed_uint64( vregs[ n ].getd( e ), 0, FPRounding_ZERO ) );
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_17 && 0x2e == bits16_10 ) // NEG <Vd>.<T>, <Vn>.<T>
                    {
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                vregs[ d ].set8( e, - (int8_t) vregs[ n ].get8( e ) );
                            else if ( 2 == ebytes )
                                vregs[ d ].set16( e, - (int16_t) vregs[ n ].get16( e ) );
                            else if ( 4 == ebytes )
                                vregs[ d ].set32( e, - (int32_t) vregs[ n ].get32( e ) );
                            else if ( 8 == ebytes )
                                vregs[ d ].set64( e, - (int64_t) vregs[ n ].get64( e ) );
                            else
                                unhandled();
                        }
                    }
                    else if ( 0x25 == bits15_10 ) // MLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                vregs[ d ].set8( e, vregs[ d ].get8( e ) - ( vregs[ n ].get8( e ) * vregs[ m ].get8( e ) ) );
                            else if ( 2 == ebytes )
                                vregs[ d ].set16( e, vregs[ d ].get16( e ) - ( vregs[ n ].get16( e ) * vregs[ m ].get16( e ) ) );
                            else if ( 4 == ebytes )
                                vregs[ d ].set32( e, vregs[ d ].get32( e ) - ( vregs[ n ].get32( e ) * vregs[ m ].get32( e ) ) );
                            else if ( 8 == ebytes )
                                vregs[ d ].set64( e, vregs[ d ].get64( e ) - ( vregs[ n ].get64( e ) * vregs[ m ].get64( e ) ) );
                            else
                                unhandled();
                        }
                    }
                    else if ( 0x6e == hi8 && ( 5 == bits23_21 || 1 == bits23_21 ) && 8 == bits20_17 && 0x32 == bits16_10 ) // FMINNMV S<d>, <Vn>.4S    ;    FMAXNMV S<d>, <Vn>.4S
                    {
                        esize = 32;
                        ebytes = 4;
                        elements = datasize / esize;

                        float result = vregs[ n ].getf( 0 );
                        for ( uint64_t e = 1; e < elements; e++ )
                            result = ( 5 == bits23_21 ) ? (float) do_fmin( result, vregs[ n ].getf( e ) ) : (float) do_fmax( result, vregs[ n ].getf( e ) );

                        zero_vreg( d );
                        vregs[ d ].setf( 0, result );
                    }
                    else if ( !bit23 && 0x3f == bits15_10 ) // FDIV <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        esize = 32ull << sz;
                        ebytes = esize / 8;
                        elements = datasize / esize;
                        if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                vregs[ d ].setf( e, (float) do_fdiv( vregs[ n ].getf( e ), vregs[ m ].getf( e ) ) );
                        else if ( 8 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                vregs[ d ].setd( e, do_fdiv( vregs[ n ].getd( e ), vregs[ m ].getd( e ) ) );
                        else
                            unhandled();
                    }
                    else if ( ( 0x1b == bits15_10 || 0x19 == bits15_10 ) ) // UMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                            {
                                uint8_t nval = (uint8_t) vregs[ n ].get8( e );
                                uint8_t mval = (uint8_t) vregs[ m ].get8( e );
                                if ( 0x19 == bits15_10 )
                                    target.set8( e, get_max( nval, mval ) );
                                else
                                    target.set8( e, get_min( nval, mval ) );
                            }
                            else if ( 2 == ebytes )
                            {
                                uint16_t nval = (uint16_t) vregs[ n ].get16( e );
                                uint16_t mval = (uint16_t) vregs[ m ].get16( e );
                                if ( 0x19 == bits15_10 )
                                    target.set16( e, get_max( nval, mval ) );
                                else
                                    target.set16( e, get_min( nval, mval ) );
                            }
                            else if ( 4 == ebytes )
                            {
                                uint32_t nval = (uint32_t) vregs[ n ].get32( e );
                                uint32_t mval = (uint32_t) vregs[ m ].get32( e );
                                if ( 0x19 == bits15_10 )
                                    target.set32( e, get_max( nval, mval ) );
                                else
                                    target.set32( e, get_min( nval, mval ) );
                            }
                            else
                                unhandled(); // no 8-byte variant exists
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 8 == bits20_17 && ( 0x6a == bits16_10 || 0x2a == bits16_10 ) ) // // UMINV <V><d>, <Vn>.<T>    ;    UMAXV <V><d>, <Vn>.<T>
                    {
                        uint8_t cur_ui8 = 0;
                        uint16_t cur_ui16 = 0;
                        uint32_t cur_ui32 = 0;

                        if ( 1 == ebytes )
                            cur_ui8 = vregs[ n ].get8( 0 );
                        else if ( 2 == ebytes )
                            cur_ui16 = vregs[ n ].get16( 0 );
                        else if ( 4 == ebytes )
                            cur_ui32 = vregs[ n ].get32( 0 );
                        else
                            unhandled(); // no 8-byte variant exists

                        for ( uint64_t e = 1; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                            {
                                uint8_t nval = vregs[ n ].get8( e );
                                if ( 0x6a == bits16_10 )
                                    cur_ui8 = get_min( cur_ui8, nval );
                                else
                                    cur_ui8 = get_max( cur_ui8, nval );
                            }
                            else if ( 2 == ebytes )
                            {
                                uint16_t nval = vregs[ n ].get16( e );
                                if ( 0x6a == bits16_10 )
                                    cur_ui16 = get_min( cur_ui16, nval );
                                else
                                    cur_ui16 = get_max( cur_ui16, nval );
                            }
                            else if ( 4 == ebytes )
                            {
                                uint32_t nval = vregs[ n ].get32( e );
                                if ( 0x6a == bits16_10 )
                                    cur_ui32 = get_min( cur_ui32, nval );
                                else
                                    cur_ui32 = get_max( cur_ui32, nval );
                            }
                        }

                        zero_vreg( d );

                        if ( 1 == ebytes )
                            vregs[ d ].set8( 0, cur_ui8 );
                        else if ( 2 == ebytes )
                            vregs[ d ].set16( 0, cur_ui16 );
                        else if ( 4 == ebytes )
                            vregs[ d ].set32( 0, cur_ui32 );
                    }
                    else if ( 4 == bits15_10 ) // UADDW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
                    {
                        datasize = 64;
                        elements = datasize / esize;
                        vec16_t target;

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, vregs[ n ].get16( e ) + (uint16_t) vregs[ m ].get8( e + ( Q ? 8 : 0 ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, vregs[ n ].get32( e ) + (uint32_t) vregs[ m ].get16( e + ( Q ? 4 : 0 ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, vregs[ n ].get64( e ) + (uint64_t) vregs[ m ].get32( e + ( Q ? 2 : 0 ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( bit23 && 0 == bits20_17 && 0x3e == bits16_10 ) // FNEG <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        esize = 32ull << sz;
                        ebytes = esize / 8;
                        elements = datasize / esize;
                        vec16_t target;
                        vec16_t & source = vregs[ n ];

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                target.setf( e, - source.getf( e ) );
                            else if ( 8 == ebytes )
                                target.setd( e, - source.getd( e ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( !bit23 && 0x35 == opcode ) // FADDP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        esize = sz ? 64 : 32;
                        ebytes = esize / 8;
                        elements = datasize / esize;
                        vec16_t target;
                        // tracer.Trace( "faddp, ebytes %llu, elements %llu\n", ebytes, elements );
                        for ( uint64_t e = 0; e < elements / 2; e++ )
                        {
                            if ( 8 == ebytes )
                                target.setd( e, do_fadd( vregs[ n ].getd( 2 * e ), vregs[ n ].getd( 2 * e + 1 ) ) );
                            else if ( 4 == ebytes )
                                target.setf( e, (float) do_fadd( vregs[ n ].getf( 2 * e ), vregs[ n ].getf( 2 * e + 1 ) ) );
                            else
                                unhandled();
                        }
                        for ( uint64_t e = 0; e < elements / 2; e++ )
                        {
                            if ( 8 == ebytes )
                                target.setd( ( elements / 2 ) + e, do_fadd( vregs[ m ].getd( 2 * e ), vregs[ m ].getd( 2 * e + 1 ) ) );
                            else if ( 4 == ebytes )
                                target.setf( ( elements / 2 ) + e, (float) do_fadd( vregs[ m ].getf( 2 * e ), vregs[ m ].getf( 2 * e + 1 ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0x11 == opcode ) // USHL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        vec16_t target;
                        vec16_t & vecm = vregs[ m ];
                        vec16_t & vecn = vregs[ n ];

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                            {
                                int8_t shift = vecm.get8( e );
                                uint8_t a = vecn.get8( e );
                                if ( shift < 0 )
                                    a >>= -shift;
                                else
                                    a <<= shift;
                                target.set8( e, a );
                            }
                            else if ( 2 == ebytes )
                            {
                                int8_t shift = (uint8_t) vecm.get16( e );
                                uint16_t a = vecn.get16( e );
                                if ( shift < 0 )
                                    a >>= -shift;
                                else
                                    a <<= shift;
                                target.set16( e, a );
                            }
                            else if ( 4 == ebytes )
                            {
                                int8_t shift = (uint8_t) vecm.get32( e );
                                uint32_t a = vecn.get32( e );
                                if ( shift < 0 )
                                    a >>= -shift;
                                else
                                    a <<= shift;
                                target.set32( e, a );
                            }
                            else if ( 8 == ebytes )
                            {
                                int8_t shift = (uint8_t) vecm.get64( e );
                                uint64_t a = vecn.get64( e );
                                if ( shift < 0 )
                                    a >>= -shift;
                                else
                                    a <<= shift;
                                target.set64( e, a );
                            }
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 8 == bits20_17 && 0xe == opcode7 ) // UADDLV <V><d>, <Vn>.<T>
                    {
                        uint64_t sum = 0;
                        vec16_t & vecn = vregs[ n ];

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                sum += vecn.get8( e );
                            else if ( 2 == ebytes )
                                sum += vecn.get16( e );
                            else if ( 4 == ebytes )
                                sum += vecn.get32( e );
                            else
                                unhandled();
                        }

                        zero_vreg( d );
                        if ( 1 == ebytes )
                            vregs[ d ].set16( 0, (uint16_t) sum );
                        else if ( 2 == ebytes )
                            vregs[ d ].set32( 0, (uint32_t) sum );
                        else if ( 4 == ebytes )
                            vregs[ d ].set64( 0, (uint64_t) sum );
                    }
                    else if ( 0x29 == opcode || 0x2b == opcode ) // UMAXP / UMINP
                    {
                        vec16_t target;
                        bool is_min = ( 0x2b == opcode );
                        vec16_t & nref = vregs[ n ];
                        vec16_t & mref = vregs[ m ];
                        if ( 1 == ebytes )
                        {
                            for ( uint64_t e = 0; e < elements; e += 2 )
                            {
                                target.set8( e / 2, is_min ? get_min( nref.get8( e ), nref.get8( e + 1 ) ) : get_max( nref.get8( e ), nref.get8( e + 1 ) ) );
                                target.set8( ( elements + e ) / 2, is_min ? get_min( mref.get8( e ), mref.get8( e + 1 ) ) : get_max( mref.get8( e ), mref.get8( e + 1 ) ) );
                            }
                        }
                        else if ( 2 == ebytes )
                        {
                            for ( uint64_t e = 0; e < elements; e += 2 )
                            {
                                target.set16( e / 2, is_min ? get_min( nref.get16( e ), nref.get16( e + 1 ) ) : get_max( nref.get16( e ), nref.get16( e + 1 ) ) );
                                target.set16( ( elements + e ) / 2, is_min ? get_min( mref.get16( e ), mref.get16( e + 1 ) ) : get_max( mref.get16( e ), mref.get16( e + 1 ) ) );
                            }
                        }
                        else if ( 4 == ebytes )
                        {
                            for ( uint64_t e = 0; e < elements; e += 2 )
                            {
                                target.set32( e / 2, is_min ? get_min( nref.get32( e ), nref.get32( e + 1 ) ) : get_max( nref.get32( e ), nref.get32( e + 1 ) ) );
                                target.set32( ( elements + e ) / 2, is_min ? get_min( mref.get32( e ), mref.get32( e + 1 ) ) : get_max( mref.get32( e ), mref.get32( e + 1 ) ) );
                            }
                        }
                        else if ( 8 == ebytes )
                        {
                            target.set64( 0, is_min ? get_min( nref.get64( 0 ), nref.get64( 1 ) ) : get_max( nref.get64( 0 ), nref.get64( 1 ) ) );
                            target.set64( 1, is_min ? get_min( mref.get64( 0 ), mref.get64( 1 ) ) : get_max( mref.get64( 0 ), mref.get64( 1 ) ) );
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 7 == opcode && 1 == opc2 ) // BSL
                    {
                        elements = Q ? 2 : 1;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            uint64_t dval = vregs[ d ].get64( e );
                            uint64_t nval = vregs[ n ].get64( e );
                            uint64_t mval = vregs[ m ].get64( e );
                            //tracer.Trace( "x: %llu, dval %#llx, nval %#llx, mval %#llx\n", x, dval, nval, mval );
                            vregs[ d ].set64( e, ( dval & nval ) | ( ( ~ dval ) & mval ) );
                        }
                    }
                    else if ( 0x37 == opcode ) // FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        esize = 32ull << sz;
                        ebytes = esize / 8;
                        elements = datasize / esize;
                        vec16_t target;
                        vec16_t & vn = vregs[ n ];
                        vec16_t & vm = vregs[ m ];
                        // tracer.Trace( "fmul sz %llu esize %llu, ebytes %llu, elements %llu\n", sz, esize, ebytes, elements );

                        if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.setf( e, (float) do_fmul( vn.getf( e ), vm.getf( e ) ) );
                        else if ( 8 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.setd( e, do_fmul( vn.getd( e ), vm.getd( e ) ) );
                        else
                            unhandled();

                        vregs[ d ] = target;
                    }
                    else
                    {
                        vec16_t target;
                        uint8_t * ptarget = (uint8_t *) &target;

                        if ( 7 == opcode ) // EOR (0 == opc2), BIT (2 == opc1), BSL (1 == opc2), and BIF (3 == opc2)
                        {
                            // EOR <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                            // BIT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                            // BSL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                            // BIF <Vd>.<T>, <Vn>.<T>, <Vm>.<T>

                            for ( uint64_t e = 0; e <= Q; e++ )
                            {
                                uint64_t dval = vregs[ d ].get64( e );
                                uint64_t nval = vregs[ n ].get64( e );
                                uint64_t mval = vregs[ m ].get64( e );
                                uint64_t result = 0;
                                if ( 0 == opc2 ) // EOR
                                    result = ( nval ^ mval );
                                else if ( 1 == opc2 ) // BSL
                                    result = ( mval ^ ( ( mval ^ nval ) & dval ) );
                                else if ( 2 == opc2 ) // BIT
                                    result = ( dval ^ ( ( dval ^ nval ) & mval ) );
                                else // BIF
                                    result = ( dval ^ ( ( dval ^ nval ) & ( ~mval ) ) );

                                target.set64( e, result );
                            }
                        }
                        else if ( 0x21 == opcode ) // SUB
                        {
                            vec16_t & nreg = vregs[ n ];
                            vec16_t & mreg = vregs[ m ];
                            for ( uint64_t e = 0; e < elements; e++ )
                            {
                                if ( 1 == ebytes )
                                    target.set8( e, nreg.get8( e ) - mreg.get8( e ) );
                                else if ( 2 == ebytes )
                                    target.set16( e, nreg.get16( e ) - mreg.get16( e ) );
                                else if ( 4 == ebytes )
                                    target.set32( e, nreg.get32( e ) - mreg.get32( e ) );
                                else if ( 8 == ebytes )
                                    target.set64( e, nreg.get64( e ) - mreg.get64( e ) );
                                else
                                    unhandled();
                            }
                        }
                        else if ( 0x30 == opcode ) // UMULL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                        {
                            datasize = 64;
                            elements = datasize / esize;
                            uint64_t eoffset = Q ? ( ( 1 == ebytes ) ? 8 : ( 2 == ebytes ) ? 4 : 2 ) : 0;
                            for ( uint64_t e = 0; e < elements; e++ )
                            {
                                if ( 1 == ebytes )
                                    target.set16( e, (uint16_t) vregs[ n ].get8( e + eoffset ) * (uint16_t) vregs[ m ].get8( e + eoffset ) );
                                else if ( 2 == ebytes )
                                    target.set32( e, (uint32_t) vregs[ n ].get16( e + eoffset ) * (uint32_t) vregs[ m ].get16( e + eoffset ) );
                                else if ( 4 == ebytes )
                                    target.set32( e, (uint64_t) vregs[ n ].get32( e + eoffset ) * (uint64_t) vregs[ m ].get32( e + eoffset ) );
                                else
                                    unhandled();
                            }
                        }
                        else if ( 0x25 == opcode ) // MLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                        {
                            vec16_t & nvec = vregs[ n ];
                            vec16_t & mvec = vregs[ m ];
                            vec16_t & dvec = vregs[ d ];

                            for ( uint64_t e = 0; e < elements; e++ )
                            {
                                if ( 1 == ebytes )
                                    target.set8( e, dvec.get8( e ) - ( nvec.get8( e ) * mvec.get8( e ) ) );
                                else if ( 2 == ebytes )
                                    target.set16( e, dvec.get16( e ) - ( nvec.get16( e ) * mvec.get16( e ) ) );
                                else if ( 4 == ebytes )
                                    target.set32( e, dvec.get32( e ) - ( nvec.get32( e ) * mvec.get32( e ) ) );
                                else if ( 8 == ebytes )
                                    target.set64( e, dvec.get64( e ) - ( nvec.get64( e ) * mvec.get64( e ) ) );
                                else
                                    unhandled();
                             }
                        }
                        else if ( 0x23 == opcode || 0x0d == opcode || 0x0f == opcode ) // vector comparisons
                        {
                            for ( uint64_t e = 0; e < elements; e++ )
                            {
                                uint64_t offset = ( e * ebytes );
                                ElementComparisonResult res = compare_vector_elements( vreg_ptr( n, offset ), vreg_ptr( m, offset ), ebytes, true );
                                bool copy_ones = ( ( 0x23 == opcode && ecr_eq == res ) ||                          // CMEQ <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                                   ( 0x0d == opcode && ecr_gt == res ) ||                          // CMHI <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                                   ( 0x0f == opcode && ( ecr_gt == res || ecr_eq == res ) ) );     // CMHS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>

                                assert( ( offset + ebytes ) <= sizeof( target ) );
                                if ( copy_ones )
                                    mcpy( ptarget + offset, &vec_ones, ebytes );
                                else
                                    mcpy( ptarget + offset, &vec_zeroes, ebytes );
                            }
                        }
                        else
                            unhandled();

                        vregs[ d ] = target;
                    }
                }
                else // !bit21
                {
                    if ( 0 == bits23_21 && !bit15 && !bit10 ) // EXT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<index>
                    {
                        uint64_t imm4 = opbits( 11, 4 ); // byte offset of first from N

                        // copy lowest from N starting with index then as much of M as fit into destination
                        vec16_t target;
                        uint64_t targeto = 0;
                        uint64_t count = Q ? 16 : 8;
                        for ( uint64_t e = imm4; e < count; e++ )
                            target.set8( targeto++, vregs[ n ].get8( e ) );
                        for ( uint64_t e = 0; targeto < count; e++ )
                            target.set8( targeto++, vregs[ m ].get8( e ) );
                        vregs[ d ] = target;
                    }
                    else if ( 0x6e == hi8 && 0 == bits23_21 && !bit15 && bit10 ) // INS <Vd>.<Ts>[<index>], <R><n>
                    {
                        uint64_t imm5 = opbits( 16, 5 );
                        uint64_t imm4 = opbits( 11, 5 );
                        uint64_t byte_width = 0;
                        uint64_t index1 = 0;
                        uint64_t index2 = 0;
                        if ( 1 & imm5 )
                        {
                            index1 = get_bits( imm5, 1, 4 );
                            index2 = imm4;
                            byte_width = 1;
                        }
                        else if ( 2 & imm5 )
                        {
                            index1 = get_bits( imm5, 2, 3 );
                            index2 = get_bits( imm4, 1, 3 );
                            byte_width = 2;
                        }
                        else if ( 4 & imm5 )
                        {
                            index1 = get_bits( imm5, 3, 2 );
                            index2 = get_bits( imm4, 2, 2 );
                            byte_width = 4;
                        }
                        else if ( 8 & imm5 )
                        {
                            index1 = get_bit( imm5, 4 );
                            index2 = get_bit( imm4, 3 );
                            byte_width = 8;
                        }

                        mcpy( vreg_ptr( d, index1 * byte_width ), vreg_ptr( n, index2 * byte_width ), byte_width );
                    }
                    else if ( 0 == size && !bit10 && !bit15 ) // EXT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<index>
                    {
                        uint64_t imm4 = opbits( 11, 4 );
                        uint64_t position = 8 * imm4;

                        if ( Q )
                        {
                            if ( 64 != position ) // not implemented
                                unhandled();

                            uint64_t nval = vregs[ n ].get64( 1 );
                            uint64_t mval = vregs[ m ].get64( 0 );
                            vregs[ d ].set64( 0, nval );
                            vregs[ d ].set64( 1, mval );
                        }
                        else
                           unhandled();
                    }
                    else
                       unhandled();
                }
                trace_vregs();
                break;
            }
            case 0x5e: // SCVTF <V><d>, <V><n>    ;    ADDP D<d>, <Vn>.2D    ;    DUP <V><d>, <Vn>.<T>[<index>]    ;    FCVTZS <V><d>, <V><n>
                       // CMGT D<d>, D<n>, D<m>   ;    CMGT D<d>, D<n>, #0   ;    ADD D<d>, D<n>, D<m>             ;    FCMLT <V><d>, <V><n>, #0.0
                       // CMEQ D<d>, D<n>, #0
            {
                uint64_t bits23_10 = opbits( 10, 14 );
                uint64_t bit23 = opbit( 23 );
                uint64_t bit21 = opbit( 21 );
                uint64_t bits23_21 = opbits( 21, 3 );
                uint64_t bits20_16 = opbits( 16, 5 );
                uint64_t bits15_10 = opbits( 10, 6 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );

                if ( 7 == bits23_21 && 0 == bits20_16 && 0x26 == bits15_10 ) // CMEQ D<d>, D<n>, #0
                {
                    vregs[ d ].set64( 0, ( 0 == vregs[ n ].get64( 0 ) ) ? ~0 : 0 );
                }
                else if ( bit23 && bit21 && 0 == bits20_16 && 0x3a == bits15_10 ) // FCMLT <V><d>, <V><n>, #0.0
                {
                    uint64_t sz = opbit( 22 );
                    if ( sz )
                        vregs[ d ].set64( 0, ( vregs[ n ].getd( 0 ) < 0.0 ) ? ~0 : 0 );
                    else
                        vregs[ d ].set32( 0, ( vregs[ n ].getf( 0 ) < 0.0 ) ? ~0 : 0 );
                    trace_vregs();
                }
                else if ( 7 == bits23_21 && 0x21 == bits15_10 ) // ADD D<d>, D<n>, D<m>
                {
                    uint64_t m = opbits( 16, 5 );
                    vregs[ d ].set64( 0, vregs[ n ].get64( 0 ) + vregs[ m ].get64( 0 ) );
                    vregs[ d ].set64( 1, 0 );
                    trace_vregs();
                }
                else if ( 7 == bits23_21 && 0xd == bits15_10 ) // CMGT D<d>, D<n>, D<m>
                {
                    uint64_t m = opbits( 16, 5 );
                    vregs[ d ].set64( 0, ( (int64_t) vregs[ n ].get64( 0 ) > (int64_t) vregs[ m ].get64( 0 ) ) ? ~0 : 0 );
                    trace_vregs();
                }
                else if ( 0x386e == bits23_10 || 0x286e == bits23_10 ) // FCVTZS <V><d>, <V><n>. round towards zero fp to signed integer
                {
                    uint64_t sz = opbit( 22 );
                    if ( sz )
                    {
                        int64_t nval = double_to_fixed_int64( vregs[ n ].getd( 0 ), 0, FPRounding_ZERO );
                        zero_vreg( d );
                        vregs[ d ].set64( 0, nval );
                    }
                    else
                    {
                        int32_t nval = double_to_fixed_int32( (double) vregs[ n ].getf( 0 ), 0, FPRounding_ZERO );
                        zero_vreg( d );
                        vregs[ d ].set32( 0, nval );
                    }
                    trace_vregs();
                }
                else if ( 0x0876 == ( bits23_10 & 0x2fff ) ) // SCVTF <Vd>.<T>, <Vn>.<T>
                {
                    uint64_t sz = opbit( 22 );
                    if ( sz )
                        vregs[ d ].setd( 0, (double) (int64_t) vregs[ n ].get64( 0 ) );
                    else
                        vregs[ d ].setf( 0, (float) (int32_t) vregs[ n ].get32( 0 ) );
                    trace_vregs();
                }
                else if ( 0x3c6e == bits23_10 ) // ADDP D<d>, <Vn>.2D
                {
                    vregs[ d ].set64( 0, vregs[ n ].get64( 0 ) + vregs[ n ].get64( 1 ) );
                    vregs[ d ].set64( 1, 0 );
                    trace_vregs();
                }
                else if ( 1 == ( bits23_10 & 0x383f ) ) // DUP <V><d>, <Vn>.<T>[<index>]   -- scalar
                {
                    uint64_t imm5 = opbits( 16, 5 );
                    uint64_t size = lowest_set_bit_nz( imm5 & 0xf );
                    uint64_t index = get_bits( imm5, size + 1, size + 2 ); // imm5:<4:size+1>
                    uint64_t esize = 8ull << size;
                    uint64_t ebytes = esize / 8;
                    vec16_t target;

                    if ( 1 == ebytes )
                        target.set8( 0, vregs[ n ].get8( index ) );
                    else if ( 2 == ebytes )
                        target.set16( 0, vregs[ n ].get16( index ) );
                    else if ( 4 == ebytes )
                        target.set32( 0, vregs[ n ].get32( index ) );
                    else if ( 8 == ebytes )
                        target.set64( 0, vregs[ n ].get64( index ) );

                    vregs[ d ] = target;
                    trace_vregs();
                }
                else
                    unhandled();
                break;
            }
            case 0x7e: // CMGE    ;    UCVTF <V><d>, <V><n>    ;    UCVTF <Hd>, <Hn>            ;    FADDP <V><d>, <Vn>.<T>    ;    FABD <V><d>, <V><n>, <V><m>
                       // FCMGE <V><d>, <V><n>, #0.0           ;    FMINNMP <V><d>, <Vn>.<T>    ;    FMAXNMP <V><d>, <Vn>.<T>
                       // CMHI D<d>, D<n>, D<m>                ;    FCVTZU <V><d>, <V><n>       ;    FCMGT <V><d>, <V><n>, <V><m>
                       // FCMGE <V><d>, <V><n>, <V><m>         ;    CMLE D<d>, D<n>, #0         ;    CMGE D<d>, D<n>, #0
            {
                uint64_t bits23_10 = opbits( 10, 14 );
                uint64_t bits23_21 = opbits( 21, 3 );
                uint64_t bits20_10 = opbits( 10, 11 );
                uint64_t bits15_10 = opbits( 10, 6 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t bit23 = opbit( 23 );
                uint64_t sz = opbit( 22 );
                uint64_t bit21 = opbit( 21 );
                uint64_t opcode = opbits( 10, 6 );

                if ( 0x3826 == bits23_10 ) // CMLE D<d>, D<n>, #0
                {
                    vregs[ d ].set64( 0, ( ( (int64_t) vregs[ n ].get64( 0 ) ) <= 0 ) ? ~0 : 0 );
                }
                else if ( !bit23 && bit21 && 0x39 == bits15_10 ) // FCMGE <V><d>, <V><n>, <V><m>
                {
                    uint64_t m = opbits( 16, 5 );
                    if ( sz )
                        vregs[ d ].set64( 0, ( vregs[ n ].getd( 0 ) >= vregs[ m ].getd( 0 ) ) ? ~0 : 0 );
                    else
                        vregs[ d ].set32( 0, ( vregs[ n ].getf( 0 ) >= vregs[ m ].getf( 0 ) ) ? 0xffffffff : 0 );
                }
                else if ( bit23 && bit21 && 0x39 == bits15_10 ) // FCMGT <V><d>, <V><n>, <V><m>
                {
                    uint64_t m = opbits( 16, 5 );
                    if ( sz )
                        vregs[ d ].set64( 0, ( vregs[ n ].getd( 0 ) > vregs[ m ].getd( 0 ) ) ? ~0 : 0 );
                    else
                        vregs[ d ].set32( 0, ( vregs[ n ].getf( 0 ) > vregs[ m ].getf( 0 ) ) ? 0xffffffff : 0 );
                }
                else if ( bit23 && bit21 && 0x6e == bits20_10 ) // FCVTZU <V><d>, <V><n>
                {
                    if ( sz )
                        vregs[ d ].set64( 0, double_to_fixed_uint64( vregs[ n ].getd( 0 ), 0, FPRounding_ZERO ) );
                    else
                        vregs[ d ].set32( 0, double_to_fixed_uint32( vregs[ n ].getf( 0 ), 0, FPRounding_ZERO ) );
                }
                else if ( 7 == bits23_21 && 0xd == bits15_10 ) // CMHI D<d>, D<n>, D<m>
                {
                    uint64_t m = opbits( 16, 5 );
                    vregs[ d ].set64( 0 , ( vregs[ n ].get64( 0 ) > vregs[ m ].get64( 0 ) ) ? ~0 : 0 );
                }
                else if ( bit21 && 0x432 == bits20_10 ) // FMINNMP <V><d>, <Vn>.<T>    ;    FMAXNMP <V><d>, <Vn>.<T>
                {
                    uint64_t esize = 32ull << sz;
                    uint64_t ebytes = esize / 8;

                    if ( 4 == ebytes )
                        vregs[ d ].setf( 0, bit23 ? (float) do_fmin( vregs[ n ].getf( 0 ), vregs[ n ].getf( 1 ) ) : (float) do_fmax( vregs[ n ].getf( 0 ), vregs[ n ].getf( 1 ) ) );
                    else if ( 8 == ebytes )
                        vregs[ d ].setd( 0, bit23 ? do_fmin( vregs[ n ].getd( 0 ), vregs[ n ].getd( 1 ) ) : do_fmax( vregs[ n ].getd( 0 ), vregs[ n ].getd( 1 ) ) );
                    else
                        unhandled();
                }
                else if ( bit23 && bit21 && 0x35 == opcode ) // FABD <V><d>, <V><n>, <V><m>     scalar single and double precision
                {
                    uint64_t m = opbits( 16, 5 );
                    if ( sz )
                    {
                        double result = fabs( vregs[ n ].getd( 0 ) - vregs[ m ].getd( 0 ) );
                        zero_vreg( d );
                        vregs[ d ].setd( 0, result );
                    }
                    else
                    {
                        float result = fabsf( vregs[ n ].getf( 0 ) - vregs[ m ].getf( 0 ) );
                        zero_vreg( d );
                        vregs[ d ].setf( 0, result );
                    }
                }
                else if ( 0x0c36 == bits23_10 || 0x1c36 == bits23_10 ) // FADDP <V><d>, <Vn>.<T>
                {
                    if ( sz )
                    {
                        double result = do_fadd( vregs[ n ].getd( 0 ), vregs[ n ].getd( 1 ) );
                        vregs[ d ].setd( 0, result );
                        vregs[ d ].set64( 1, 0 );
                    }
                    else
                    {
                        float result = (float) do_fadd( vregs[ n ].getf( 0 ), vregs[ n ].getf( 1 ) );
                        zero_vreg( d );
                        vregs[ d ].setf( 0, result );
                    }
                    trace_vregs();
                }
                else if ( 0x3822 == bits23_10 ) // CMGE D<d>, D<n>, #0
                {
                    vregs[ d ].set64( 0, ( ( (int64_t) vregs[ n ].get64( 0 ) >= 0 ) ) ? ~0 : 0 );
                }
                else if ( 0x0876 == ( bits23_10 & 0x2fff ) ) // UCVTF <V><d>, <V><n>
                {
                    if ( sz )
                        vregs[ d ].setd( 0, (double) vregs[ n ].get64( 0 ) );
                    else
                        vregs[ d ].setf( 0, (float) vregs[ n ].get32( 0 ) );
                }
                else if ( 0x2832 == bits23_10 || 0x3832 == bits23_10 ) // FCMGE <V><d>, <V><n>, #0.0
                {
                    if ( sz )
                        vregs[ d ].set64( 0, ( vregs[ n ].getd( 0 ) >= 0.0 ) ? ~0 : 0 );
                    else
                        vregs[ d ].set32( 0, ( vregs[ n ].getf( 0 ) >= 0.0f ) ? ~0 : 0 );
                }
                else
                    unhandled();
                break;
            }
            case 0x0e: case 0x4e: // DUP <Vd>.<T>, <Vn>.<Ts>[<index>]    ;    DUP <Vd>.<T>, <R><n>    ;             CMEQ <Vd>.<T>, <Vn>.<T>, #0    ;    ADDP <Vd>.<T>, <V
                                  // AND <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMOV <Wd>, <Vn>.<Ts>[<index>]    ;    UMOV <Xd>, <Vn>.D[<index>]     ;    CNT <Vd>.<T>, <Vn>.<T>
                                  // AND <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UMOV <Wd>, <Vn>.<Ts>[<index>]    ;    UMOV <Xd>, <Vn>.D[<index>]     ;    ADDV <V><d>, <Vn>.<T>
                                  // XTN{2} <Vd>.<Tb>, <Vn>.<Ta>         ;    UZP1 <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   UZP2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // SMOV <Wd>, <Vn>.<Ts>[<index>]       ;    SMOV <Xd>, <Vn>.<Ts>[<index>]    ;    INS <Vd>.<Ts>[<index>], <R><n> ;    CMGT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // SCVTF <Vd>.<T>, <Vn>.<T>            ;    FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<T>;    FADD <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // TRN1 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    TRN2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   TBL <Vd>.<Ta>, { <Vn>.16B }, <Vm>.<Ta> ; TBL <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B, <Vn+2>.16B, <Vn+3>.16B }, <Vm>.<Ta>
                                  // ZIP1 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    ZIP2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   SMULL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                                  // MLA <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    MUL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>  ;   CMLT <Vd>.<T>, <Vn>.<T>, #0    ;    REV64 <Vd>.<T>, <Vn>.<T>
                                  // BIC <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    FMLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   FSUB <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // SMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    SMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;   SMINV <V><d>, <Vn>.<T>         ;    SMAXV <V><d>, <Vn>.<T>
                                  // FMINNM <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;    FMAXNM <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ; FCVTN{2} <Vd>.<Tb>, <Vn>.<Ta>  ;    FCVTZS <Vd>.<T>, <Vn>.<T>
                                  // ORN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    FCVTL{2} <Vd>.<Ta>, <Vn>.<Tb>     ;   SSHL <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ; SADDW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
                                  // CMGE <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    ABS <Vd>.<T>, <Vn>.<T>            ;   SSUBW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
                                  // FCMLT <Vd>.<T>, <Vn>.<T>, #0.0      ;    FABS <Vd>.<T>, <Vn>.<T>           ;   SMLAL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                                  // SADDL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb> ; SADDLP <Vd>.<Ta>, <Vn>.<Tb>     ;   SADDLV <V><d>, <Vn>.<T>        ;    SADALP <Vd>.<Ta>, <Vn>.<Tb>
                                  // SQXTN{2} <Vd>.<Tb>, <Vn>.<Ta>       ;    CMGT <Vd>.<T>, <Vn>.<T>, #0       ;   CMTST <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                                  // FMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>   ;    FCMEQ <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;  REV16 <Vd>.<T>, <Vn>.<T>
            {
                uint64_t Q = opbit( 30 );
                uint64_t imm5 = opbits( 16, 5 );
                uint64_t bit15 = opbit( 15 );
                uint64_t bits14_11 = opbits( 11, 4 );
                uint64_t bit10 = opbit( 10 );
                uint64_t bit21 = opbit( 21 );
                uint64_t bit23 = opbit( 23 );
                uint64_t bits23_21 = opbits( 21, 3 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );
                uint64_t datasize = 64ull << Q;
                uint64_t bits20_16 = opbits( 16, 5 );
                uint64_t bits14_10 = opbits( 10, 5 );
                uint64_t bits12_10 = opbits( 10, 3 );
                uint64_t bits15_10 = opbits( 10, 6 );

                if ( bit21 )
                {
                    if ( 0 == bits20_16 && 6 == bits15_10 ) // REV16 <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        if ( 0 != size )
                            unhandled();

                        uint64_t csize = 16;
                        uint64_t containers = datasize / csize;
                        vec16_t target;
                        for ( uint64_t c = 0; c < containers; c++ )
                            target.set16( c, flip_endian16( vregs[ n ].get16( c ) ) );
                        vregs[ d ] = target;
                    }
                    else if ( !bit23 && 0x39 == bits15_10 ) // FCMEQ <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 32 << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                target.set32( e, ( vregs[ n ].getf( e ) == vregs[ m ].getf( e ) ) ? ~0 : 0 );
                            else if ( 8 == ebytes )
                                target.set64( e, ( vregs[ n ].getd( e ) == vregs[ m ].getd( e ) ) ? ~0 : 0 );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0x3d == bits15_10 ) // FMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    FMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        bool fpcr_ah = get_bit( fpcr, 1 );
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                            {
                                float result = 0;
                                float nval = vregs[ n ].getf( e );
                                float mval = vregs[ m ].getf( e );
                                bool n_nan = isnan( nval );
                                bool m_nan = isnan( mval );

                                if ( n_nan || m_nan )
                                {
                                    if ( fpcr_ah )
                                        result = mval;
                                    else
                                        result = (float) MY_NAN;
                                }
                                else
                                {
                                    if ( bit23 )
                                        result = (float) do_fmin( nval, mval );
                                    else
                                        result = (float) do_fmax( nval, mval );
                                }
                                vregs[ d ].setf( e, result );
                            }
                            else if ( 8 == ebytes )
                            {
                                double result = 0;
                                double nval = vregs[ n ].getd( e );
                                double mval = vregs[ m ].getd( e );
                                bool n_nan = isnan( nval );
                                bool m_nan = isnan( mval );

                                if ( n_nan || m_nan )
                                {
                                    if ( fpcr_ah )
                                        result = mval;
                                    else
                                        result = MY_NAN;
                                }
                                else
                                {
                                    if ( bit23 )
                                        result = do_fmin( nval, mval );
                                    else
                                        result = do_fmax( nval, mval );
                                }
                                vregs[ d ].setd( e, result );
                            }
                            else
                                unhandled();
                        }
                    }
                    else if ( 0x23 == bits15_10 ) // CMTST <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t m = opbits( 16, 5 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                vregs[ d ].set8( e, ( vregs[ n ].get8( e ) & vregs[ m ].get8( e ) ) ? 0xff : 0 );
                            else if ( 2 == ebytes )
                                vregs[ d ].set16( e, ( vregs[ n ].get16( e ) & vregs[ m ].get16( e ) ) ? 0xffff : 0 );
                            else if ( 4 == ebytes )
                                vregs[ d ].set32( e, ( vregs[ n ].get32( e ) & vregs[ m ].get32( e ) ) ? 0xffffffff : 0 );
                            else if ( 8 == ebytes )
                                vregs[ d ].set64( e, ( vregs[ n ].get64( e ) & vregs[ m ].get64( e ) ) ? ~0 : 0 );
                            else
                                unhandled();
                        }
                    }
                    else if ( bit21 && 0 == bits20_16 && 0x22 == bits15_10 ) // CMGT <Vd>.<T>, <Vn>.<T>, #0
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                            {
                                if ( ( (int8_t) vregs[ n ].get8( e ) ) > 0 )
                                    target.set8( e, 0xff );
                            }
                            else if ( 2 == ebytes )
                            {
                                if ( ( (int16_t) vregs[ n ].get16( e ) ) > 0 )
                                    target.set16( e, 0xffff );
                            }
                            else if ( 4 == ebytes )
                            {
                                if ( ( (int32_t) vregs[ n ].get32( e ) ) > 0 )
                                    target.set32( e, 0xffffffff );
                            }
                            else if ( 8 == ebytes )
                            {
                                if ( ( (int64_t) vregs[ n ].get64( e ) ) > 0 )
                                    target.set64( e, ~0ull );
                            }
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 1 == bits20_16 && 0x12 == bits15_10 ) // SQXTN{2} <Vd>.<Tb>, <Vn>.<Ta>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        datasize = 64;
                        uint64_t elements = datasize / esize;
                        vec16_t target = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                            {
                                int16_t i16 = vregs[ n ].get16( e );
                                int8_t i8 = ( i16 < INT8_MIN ) ? INT8_MIN : ( i16 > INT8_MAX ) ? INT8_MAX : (int8_t) i16;
                                if ( Q )
                                    target.set8( 8 + e, i8 );
                                else
                                {
                                    target.set8( e, i8 );
                                    target.set8( 8 + e, 0 );
                                }
                            }
                            else if ( 2 == ebytes )
                            {
                                int32_t i32 = vregs[ n ].get32( e );
                                int16_t i16 = ( i32 < INT16_MIN ) ? INT16_MIN : ( i32 > INT16_MAX ) ? INT16_MAX : (int16_t) i32;
                                if ( Q )
                                    target.set16( 4 + e, i16 );
                                else
                                {
                                    target.set16( e, i16 );
                                    target.set16( 4 + e, 0 );
                                }
                            }
                            else if ( 4 == ebytes )
                            {
                                int64_t i64 = vregs[ n ].get64( e );
                                int32_t i32 = ( i64 < INT32_MIN ) ? INT32_MIN : ( i64 > INT32_MAX ) ? INT32_MAX : (int32_t) i64;
                                if ( Q )
                                    target.set32( 2 + e, i32 );
                                else
                                {
                                    target.set32( e, i32 );
                                    target.set32( 2 + e, 0 );
                                }
                            }
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_16 && 0x1a == bits15_10 ) //SADALP <Vd>.<Ta>, <Vn>.<Tb>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        datasize = 64 << Q;
                        uint64_t elements = datasize / ( 2 * esize );
                        vec16_t target = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, target.get16( e ) + ( (int16_t) sign_extend( vregs[ n ].get8( 2 * e ), 7 ) +
                                                                       (int16_t) sign_extend( vregs[ n ].get8( 2 * e + 1 ), 7 ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, target.get32( e ) + ( (int32_t) sign_extend( vregs[ n ].get16( 2 * e ), 15 ) +
                                                                       (int32_t) sign_extend( vregs[ n ].get16( 2 * e + 1 ), 15 ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, target.get64( e ) + ( (int64_t) sign_extend( vregs[ n ].get32( 2 * e ), 31 ) +
                                                                       (int64_t) sign_extend( vregs[ n ].get32( 2 * e + 1 ), 31 ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0x10 == bits20_16 && 0x0e == bits15_10 ) //SADDLV <V><d>, <Vn>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        datasize = 64 << Q;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( 0, target.get16( 0 ) + (int16_t) sign_extend( vregs[ n ].get8( e ), 7 ) );
                            else if ( 2 == ebytes )
                                target.set32( 0, target.get32( 0 ) + (int32_t) sign_extend( vregs[ n ].get16( e ), 15 ) );
                            else if ( 4 == ebytes )
                                target.set64( 0, target.get64( 0 ) + (int64_t) sign_extend( vregs[ n ].get32( e ), 31 ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_16 && 0x0a == bits15_10 ) // SADDLP <Vd>.<Ta>, <Vn>.<Tb>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        datasize = 64 << Q;
                        uint64_t elements = datasize / ( 2 * esize );
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, ( (int16_t) sign_extend( vregs[ n ].get8( 2 * e ), 7 ) +
                                                   (int16_t) sign_extend( vregs[ n ].get8( 2 * e + 1 ), 7 ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, ( (int32_t) sign_extend( vregs[ n ].get16( 2 * e ), 15 ) +
                                                   (int32_t) sign_extend( vregs[ n ].get16( 2 * e + 1 ), 15 ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, ( (int64_t) sign_extend( vregs[ n ].get32( 2 * e ), 31 ) +
                                                   (int64_t) sign_extend( vregs[ n ].get32( 2 * e + 1 ), 31 ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits15_10 ) // SADDL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t m = opbits( 16, 5 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        datasize = 64;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, ( (int16_t) sign_extend( vregs[ n ].get8( ( Q ? 8 : 0 ) + e ), 7 ) +
                                                   (int16_t) sign_extend( vregs[ m ].get8( ( Q ? 8 : 0 ) + e ), 7 ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, ( (int32_t) sign_extend( vregs[ n ].get16( ( Q ? 4 : 0 ) + e ), 15 ) +
                                                   (int32_t) sign_extend( vregs[ m ].get16( ( Q ? 4 : 0 ) + e ), 15 ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, ( (int64_t) sign_extend( vregs[ n ].get32( ( Q ? 2 : 0 ) + e ), 31 ) +
                                                   (int64_t) sign_extend( vregs[ m ].get32( ( Q ? 2 : 0 ) + e ), 31 ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0x20 == bits15_10 ) // SMLAL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t m = opbits( 16, 5 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        datasize = 64;
                        uint64_t elements = datasize / esize;
                        vec16_t target = vregs[ d ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, target.get16( e ) + ( (int16_t) sign_extend( vregs[ n ].get8( ( Q ? 8 : 0 ) + e ), 7 ) *
                                                                       (int16_t) sign_extend( vregs[ m ].get8( ( Q ? 8 : 0 ) + e ), 7 ) ) );
                            else if ( 2 == ebytes )
                                target.set32( e, target.get32( e ) + ( (int32_t) sign_extend( vregs[ n ].get16( ( Q ? 4 : 0 ) + e ), 15 ) *
                                                                       (int32_t) sign_extend( vregs[ m ].get16( ( Q ? 4 : 0 ) + e ), 15 ) ) );
                            else if ( 4 == ebytes )
                                target.set64( e, target.get64( e ) + ( (int64_t) sign_extend( vregs[ n ].get32( ( Q ? 2 : 0 ) + e ), 31 ) *
                                                                       (int64_t) sign_extend( vregs[ m ].get32( ( Q ? 2 : 0 ) + e ), 31 ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( bit23 && 0 == bits20_16 && 0x3e == bits15_10 ) // FABS <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                vregs[ d ].setf( e, do_abs( vregs[ n ].getf( e ) ) );
                            else if ( 8 == ebytes )
                                vregs[ d ].setd( e, do_abs( vregs[ n ].getd( e ) ) );
                            else
                                unhandled();
                        }
                    }
                    else if ( bit23 && 0 == bits20_16 && 0x3a == bits15_10 ) // FCMLT <Vd>.<T>, <Vn>.<T>, #0.0
                    {
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                vregs[ d ].set32( e, ( vregs[ n ].getf( e ) < 0 ) ? ~0 : 0 );
                            else if ( 8 == ebytes )
                                vregs[ d ].set64( e, ( vregs[ n ].getd( e ) < 0 ) ? ~0 : 0 );
                            else
                                unhandled();
                        }
                    }
                    else if ( 0xc == bits15_10 ) // SSUBW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        datasize = 64;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, (int16_t) vregs[ n ].get16( e ) - (int16_t) sign_extend( vregs[ m ].get8( ( Q ? 8 : 0 ) + e ), 7 ) );
                            else if ( 2 == ebytes )
                                target.set32( e, (int32_t) vregs[ n ].get32( e ) - (int32_t) sign_extend( vregs[ m ].get16( ( Q ? 4 : 0 ) + e ), 15 ) );
                            else if ( 4 == ebytes )
                                target.set64( e, (int64_t) vregs[ n ].get64( e ) - (int64_t) sign_extend( vregs[ m ].get32( ( Q ? 2 : 0 ) + e ), 31 ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_16 && 0x2e == bits15_10 ) // ABS <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                vregs[ d ].set8( e, do_abs( (int8_t) vregs[ n ].get8( e ) ) );
                            else if ( 2 == ebytes )
                                vregs[ d ].set16( e, do_abs( (int16_t) vregs[ n ].get16( e ) ) );
                            else if ( 4 == ebytes )
                                vregs[ d ].set32( e, do_abs( (int32_t) vregs[ n ].get32( e ) ) );
                            else if ( 8 == ebytes )
                                vregs[ d ].set64( e, do_abs( (int64_t) vregs[ n ].get64( e ) ) );
                            else
                                unhandled();
                        }
                    }
                    else if ( 4 == bits15_10 ) // SADDW{2} <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                target.set16( e, (int16_t) vregs[ n ].get16( e ) + (int16_t) sign_extend( vregs[ m ].get8( ( Q ? 8 : 0 ) + e ), 7 ) );
                            else if ( 2 == ebytes )
                                target.set32( e, (int32_t) vregs[ n ].get32( e ) + (int32_t) sign_extend( vregs[ m ].get16( ( Q ? 4 : 0 ) + e ), 15 ) );
                            else if ( 4 == ebytes )
                                target.set64( e, (int64_t) vregs[ n ].get64( e ) + (int64_t) sign_extend( vregs[ m ].get32( ( Q ? 2 : 0 ) + e ), 31 ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0x11 == bits15_10 ) // SSHL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                            {
                                int8_t shift = (int8_t) vregs[ m ].get8( e );
                                if ( shift >= 0 )
                                    target.set8( e, vregs[ n ].get8( e ) << shift );
                                else
                                    target.set8( e, ( (int8_t) vregs[ n ].get8( e ) ) >> ( -shift ) );
                            }
                            else if ( 2 == ebytes )
                            {
                                int8_t shift = (int8_t) ( 0xff & vregs[ m ].get16( e ) );
                                if ( shift >= 0 )
                                    target.set16( e, vregs[ n ].get16( e ) << shift );
                                else
                                    target.set16( e, ( (int16_t) vregs[ n ].get16( e ) ) >> ( -shift ) );
                            }
                            else if ( 4 == ebytes )
                            {
                                int8_t shift = (int8_t) ( 0xff & vregs[ m ].get32( e ) );
                                if ( shift >= 0 )
                                    target.set32( e, vregs[ n ].get32( e ) << shift );
                                else
                                    target.set32( e, ( (int32_t) vregs[ n ].get32( e ) ) >> ( -shift ) );
                            }
                            else if ( 8 == ebytes )
                            {
                                int8_t shift = (int8_t) ( 0xff & vregs[ m ].get64( e ) );
                                if ( shift >= 0 )
                                    target.set64( e, vregs[ n ].get64( e ) << shift );
                                else
                                    target.set64( e, ( (int64_t) vregs[ n ].get64( e ) ) >> ( -shift ) );
                            }
                        }

                        vregs[ d ] = target;
                    }
                    else if ( !bit23 && 1 == bits20_16 && 0x1e == bits15_10 ) // FCVTL{2} <Vd>.<Ta>, <Vn>.<Tb>
                    {
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 16ull << sz;
                        uint64_t ebytes = esize / 8;
                        datasize = 64;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                target.setd( e, round_double( (double) vregs[ n ].getf( e + ( Q ? 2 : 0 ) ), fp_decode_rmode( get_bits( fpcr, 22, 2 ) ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 7 == bits23_21 && 7 == bits15_10 ) // ORN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        vregs[ d ].set64( 0, vregs[ n ].get64( 0 ) | ( ~ vregs[ m ].get64( 0 ) ) );
                        if ( Q )
                            vregs[ d ].set64( 1, vregs[ n ].get64( 1 ) | ( ~ vregs[ m ].get64( 1 ) ) );
                        else
                            vregs[ d ].set64( 1, 0 );
                    }
                    else if ( bit23 && 1 == bits20_16 && 0x2e == bits15_10 ) // FCVTZS <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                target.set32( e, double_to_fixed_int32( (double) vregs[ n ].getf( e ), 0, FPRounding_ZERO ) );
                            else if ( 8 == ebytes )
                                target.set64( e, double_to_fixed_int64( vregs[ n ].getd( e ), 0, FPRounding_ZERO ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( !bit23 && 1 == bits20_16 && 0x1a == bits15_10 ) // FCVTN{2} <Vd>.<Tb>, <Vn>.<Ta>
                    {
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 16ull << sz;
                        datasize = 64;
                        uint64_t elements = datasize / esize;
                        vec16_t target = Q ? vregs[ d ] : vec_zeroes;

                        if ( 0 == sz )
                            unhandled(); // H floats not supported

                        for ( uint64_t e = 0; e < elements; e++ )
                            target.setf( ( Q ? 2 : 0 ) + e, (float) vregs[ n ].getd( e ) );
                        vregs[ d ] = target;
                    }
                    else if ( 0x31 == bits15_10 ) // FMINNM <Vd>.<T>, <Vn>.<T>, <Vm>.<T> ;    FMAXNM <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                vregs[ d ].setf( e, bit23 ? (float) do_fmin( vregs[ n ].getf( e ), vregs[ m ].getf( e ) ) : (float) do_fmax( vregs[ n ].getf( e ), vregs[ m ].getf( e ) ) );
                            else if ( 8 == ebytes )
                                vregs[ d ].setd( e, bit23 ? do_fmin( vregs[ n ].getd( e ), vregs[ m ].getd( e ) ) : do_fmax( vregs[ n ].getd( e ), vregs[ m ].getd( e ) ) );
                            else
                                unhandled();
                        }
                        trace_vregs();
                    }
                    else if ( ( 0x11 == bits20_16 || 0x10 == bits20_16 ) && 0x2a == bits15_10 ) // SMINV <V><d>, <Vn>.<T>         ;    SMAXV <V><d>, <Vn>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        int8_t cur_i8 = 0;
                        int16_t cur_i16 = 0;
                        int32_t cur_i32 = 0;

                        if ( 1 == ebytes )
                            cur_i8 = vregs[ n ].get8( 0 );
                        else if ( 2 == ebytes )
                            cur_i16 = vregs[ n ].get16( 0 );
                        else if ( 4 == ebytes )
                            cur_i32 = vregs[ n ].get32( 0 );
                        else
                            unhandled(); // no 8-byte variant exists

                        for ( uint64_t e = 1; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                            {
                                int8_t nval = vregs[ n ].get8( e );
                                if ( 0x11 == bits20_16 )
                                    cur_i8 = get_min( cur_i8, nval );
                                else
                                    cur_i8 = get_max( cur_i8, nval );
                            }
                            else if ( 2 == ebytes )
                            {
                                int16_t nval = vregs[ n ].get16( e );
                                if ( 0x11 == bits20_16 )
                                    cur_i16 = get_min( cur_i16, nval );
                                else
                                    cur_i16 = get_max( cur_i16, nval );
                            }
                            else if ( 4 == ebytes )
                            {
                                int32_t nval = vregs[ n ].get32( e );
                                if ( 0x11 == bits20_16 )
                                    cur_i32 = get_min( cur_i32, nval );
                                else
                                    cur_i32 = get_max( cur_i32, nval );
                            }
                        }

                        zero_vreg( d );

                        if ( 1 == ebytes )
                            vregs[ d ].set8( 0, cur_i8 );
                        else if ( 2 == ebytes )
                            vregs[ d ].set16( 0, cur_i16 );
                        else if ( 4 == ebytes )
                            vregs[ d ].set32( 0, cur_i32 );
                    }
                    else if ( 0x19 == bits15_10 || 0x1b == bits15_10 ) // SMAX <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    SMIN <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                            {
                                int8_t nval = (int8_t) vregs[ n ].get8( e );
                                int8_t mval = (int8_t) vregs[ m ].get8( e );
                                if ( 0x19 == bits15_10 )
                                    target.set8( e, get_max( nval, mval ) );
                                else
                                    target.set8( e, get_min( nval, mval ) );
                            }
                            else if ( 2 == ebytes )
                            {
                                int16_t nval = (int16_t) vregs[ n ].get16( e );
                                int16_t mval = (int16_t) vregs[ m ].get16( e );
                                if ( 0x19 == bits15_10 )
                                    target.set16( e, get_max( nval, mval ) );
                                else
                                    target.set16( e, get_min( nval, mval ) );
                            }
                            else if ( 4 == ebytes )
                            {
                                int32_t nval = (int32_t) vregs[ n ].get32( e );
                                int32_t mval = (int32_t) vregs[ m ].get32( e );
                                if ( 0x19 == bits15_10 )
                                    target.set32( e, get_max( nval, mval ) );
                                else
                                    target.set32( e, get_min( nval, mval ) );
                            }
                            else
                                unhandled(); // no 8-byte variant exists
                        }
                        vregs[ d ] = target;
                    }
                    else if ( bit23 && 0x35 == bits15_10 ) // FSUB <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;

                        if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                vregs[ d ].setf( e, (float) do_fsub( vregs[ n ].getf( e ), vregs[ m ].getf( e ) ) );
                        else if ( 8 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                vregs[ d ].setd( e, do_fsub( vregs[ n ].getd( e ), vregs[ m ].getd( e ) ) );
                        else
                            unhandled();
                    }
                    else if ( bit23 && 0x33 == bits15_10 ) // FMLS <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;

                        if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                vregs[ d ].setf( e, (float) do_fsub( vregs[ d ].getf( e ), do_fmul( vregs[ n ].getf( e ), vregs[ m ].getf( e ) ) ) );
                        else if ( 8 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                vregs[ d ].setd( e, do_fsub( vregs[ d ].getd( e ), do_fmul( vregs[ n ].getd( e ), vregs[ m ].getd( e ) ) ) );
                        else
                            unhandled();
                    }
                    else if ( 3 == bits23_21 && 7 == bits15_10 ) // BIC <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        vec16_t target;

                        target.set64( 0, ( vregs[ n ].get64( 0 ) & ( ~ ( vregs[ m ].get64( 0 ) ) ) ) );
                        if ( Q )
                            target.set64( 1, ( vregs[ n ].get64( 1 ) & ( ~ ( vregs[ m ].get64( 1 ) ) ) ) );

                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_16 && !bit15 && 2 == bits14_10 ) // REV64 <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t csize = 64ull;
                        uint64_t containers = datasize / csize;
                        vec16_t target;

                        for ( uint64_t c = 0; c < containers; c++ )
                            target.set64( c, reverse_bytes( vregs[ n ].get64( c ), esize ) );

                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits20_16 && bit15 && 0xa == bits14_10 ) // CMLT <Vd>.<T>, <Vn>.<T>, #0
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        vec16_t & vn = vregs[ n ];

                        if ( 1 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set8( e, (uint8_t) ( ( 0x80 & vn.get8( e ) ) ? ~0 : 0 ) );
                        else if ( 2 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set16( e, (uint16_t) ( ( 0x8000 & vn.get16( e ) ) ? ~0 : 0 ) );
                        else if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set32( e, (uint32_t) ( ( 0x80000000 & vn.get32( e ) ) ? ~0 : 0 ) );
                        else if ( 8 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set64( e, (uint64_t) ( ( 0x8000000000000000 & vn.get64( e ) ) ? ~0 : 0 ) );
                        else
                            unhandled();

                        vregs[ d ] = target;
                    }
                    else if ( bit15 && ( 7 == bits14_10 || 5 == bits14_10 ) ) // MUL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    MLA <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        bool accumulate = ( 5 == bits14_10 ); // mla

                        vec16_t & vn = vregs[ n ];
                        vec16_t & vm = vregs[ m ];
                        vec16_t & vd = vregs[ d ];

                        if ( 1 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set8( e, (uint8_t) ( ( vn.get8( e ) * vm.get8( e ) ) + ( accumulate ? vd.get8( e ) : 0 ) ) );
                        else if ( 2 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set16( e, (uint16_t) ( ( vn.get16( e ) * vm.get16( e ) ) + ( accumulate ? vd.get16( e ) : 0 ) ) );
                        else if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set32( e, (uint32_t) ( ( vn.get32( e ) * vm.get32( e ) ) + ( accumulate ? vd.get32( e ) : 0 ) ) );
                        else if ( 8 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set64( e, (uint64_t) ( ( vn.get64( e ) * vm.get64( e ) ) + ( accumulate ? vd.get64( e ) : 0 ) ) );
                        else
                            unhandled();

                        vregs[ d ] = target;
                    }
                    else if ( bit15 && 8 == bits14_11 && !bit10 ) // SMULL{2} <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        assert( ebytes <= 4 );
                        datasize = 64;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        uint64_t in_offset = Q ? ( ( 1 == ebytes ) ? 8 : ( 2 == ebytes ) ? 4 : 2 ) : 0;
                        vec16_t & vn = vregs[ n ];
                        vec16_t & vm = vregs[ m ];

                        if ( 1 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set16( e, (int16_t) (int8_t) vn.get8( e + in_offset ) * (int16_t) (int8_t) vm.get8( e + in_offset ) );
                        else if ( 2 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set32( e, (int32_t) (int16_t) vn.get16( e + in_offset ) * (int32_t) (int16_t) vm.get16( e + in_offset ) );
                        else if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set64( e, (int64_t) (int32_t) vn.get32( e + in_offset ) * (int64_t) (int32_t) vm.get32( e + in_offset ) );
                        else
                            unhandled();

                        vregs[ d ] = target;
                    }
                    else if ( !bit23 && bit15 && 0xa == bits14_11 && bit10 ) // FADD <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        uint64_t m = opbits( 16, 5 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        vec16_t & vn = vregs[ n ];
                        vec16_t & vm = vregs[ m ];

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 8 == ebytes )
                                target.setd( e, do_fadd( vn.getd( e ), vm.getd( e ) ) );
                            else if ( 4 == ebytes )
                                target.setf( e, (float) do_fadd( vn.getf( e ), vm.getf( e ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( !bit23 && bit15 && 9 == bits14_11 && bit10 ) // FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        uint64_t m = opbits( 16, 5 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        vec16_t & vn = vregs[ n ];
                        vec16_t & vm = vregs[ m ];
                        vec16_t & vd = vregs[ d ];

                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 8 == ebytes )
                                target.setd( e, do_fadd( do_fmul( vn.getd( e ), vm.getd( e ) ), vd.getd( e ) ) );
                            else if ( 4 == ebytes )
                                target.setf( e, (float) do_fadd( do_fmul( vn.getf( e ), vm.getf( e ) ), (double) vd.getf( e ) ) );
                            else
                                unhandled();
                        }
                        vregs[ d ] = target;
                    }
                    else if ( !bit23 && 1 == bits20_16 && bit15 && 0x16 == bits14_10 ) // SCVTF <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t sz = opbit( 22 );
                        uint64_t esize = 32ull << sz;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 4 == ebytes )
                                vregs[ d ].setf( e, (float) (int32_t) vregs[ n ].get32( e ) );
                            else if ( 8 == ebytes )
                                vregs[ d ].setd( e, (double) (int64_t) vregs[ n ].get64( e ) );
                        }
                    }
                    else if ( 1 == bits23_21 && !bit15 && 3 == bits14_11 && bit10 ) // AND <Vd>.<T>,
                    {
                        uint64_t m = imm5;
                        uint64_t lo = vregs[ n ].get64( 0 ) & vregs[ m ].get64( 0 );
                        uint64_t hi = 0;
                        if ( Q )
                            hi = vregs[ n ].get64( 1 ) & vregs[ m ].get64( 1 );
                        vregs[ d ].set64( 0, lo );
                        vregs[ d ].set64( 1, hi );
                    }
                    else if ( 5 == bits23_21 && !bit15 && 3 == bits14_11 && bit10 ) // ORR <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = imm5;
                        uint64_t lo = vregs[ n ].get64( 0 ) | vregs[ m ].get64( 0 );
                        uint64_t hi = 0;
                        if ( Q )
                            hi = vregs[ n ].get64( 1 ) | vregs[ m ].get64( 1 );
                        vregs[ d ].set64( 0, lo );
                        vregs[ d ].set64( 1, hi );
                    }
                    else if ( bit15 && 3 == bits14_11 && !bit10 && 0 == bits20_16 )  // CMEQ <Vd>.<T>, <Vn>.<T>, #0
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t & dref = vregs[ d ];
                        vec16_t & nref = vregs[ n ];
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                dref.set8( e, ( 0 == nref.get8( e ) ) ? ~0 : 0 );
                            else if ( 2 == ebytes )
                                dref.set16( e, ( 0 == nref.get16( e ) ) ? ~0 : 0 );
                            else if ( 4 == ebytes )
                                dref.set32( e, ( 0 == nref.get32( e ) ) ? ~0 : 0 );
                            else
                                dref.set64( e, ( 0 == nref.get64( e ) ) ? ~0 : 0 );
                        }
                    }
                    else if ( !bit15 && ( 7 == bits14_11 || 6 == bits14_11 ) && bit10 ) // CMGE <Vd>.<T>, <Vn>.<T>, <Vm>.<T>  ;  CMGT <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t & dref = vregs[ d ];
                        vec16_t & nref = vregs[ n ];
                        vec16_t & mref = vregs[ m ];
                        bool is_gt = ( 6 == bits14_11 );
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            if ( 1 == ebytes )
                                dref.set8( e, is_gt ? ( (int8_t) nref.get8( e ) > (int8_t) mref.get8( e ) ? ~0 : 0 ) : ( (int8_t) nref.get8( e ) >= (int8_t) mref.get8( e ) ? ~0 : 0 ) );
                            else if ( 2 == ebytes )
                                dref.set16( e, is_gt ? ( (int16_t) nref.get16( e ) > (int16_t) mref.get16( e ) ? ~0 : 0 ) : ( (int16_t) nref.get16( e ) >= (int16_t) mref.get16( e ) ? ~0 : 0 ) );
                            else if ( 4 == ebytes )
                                dref.set32( e, is_gt ? ( (int32_t) nref.get32( e ) > (int32_t) mref.get32( e ) ? ~0 : 0 ) : ( (int32_t) nref.get32( e ) >= (int32_t) mref.get32( e ) ? ~0 : 0 ) );
                            else
                                dref.set64( e, is_gt ? ( (int64_t) nref.get64( e ) > (int64_t) mref.get64( e ) ? ~0 : 0 ) : ( (int64_t) nref.get64( e ) >= (int64_t) mref.get64( e ) ? ~0 : 0 ) );
                        }
                    }
                    else if ( bit15 && 7 == bits14_11 && bit10 ) // ADDP <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t m = opbits( 16, 5 );
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        //tracer.Trace( "elements: %llu, ebytes %llu\n", elements, ebytes );

                        vec16_t target;
                        vec16_t & nref = vregs[ n ];
                        vec16_t & mref = vregs[ m ];
                        if ( 1 == ebytes )
                        {
                            for ( uint64_t e = 0; e < elements; e += 2 )
                            {
                                target.set8( e / 2, nref.get8( e ) + nref.get8( e + 1 ) );
                                target.set8( ( elements + e ) / 2, mref.get8( e ) + mref.get8( e + 1 ) );
                            }
                        }
                        else if ( 2 == ebytes )
                        {
                            for ( uint64_t e = 0; e < elements; e += 2 )
                            {
                                target.set16( e / 2, nref.get16( e ) + nref.get16( e + 1 ) );
                                target.set16( ( elements + e ) / 2, mref.get16( e ) + mref.get16( e + 1 ) );
                            }
                        }
                        else if ( 4 == ebytes )
                        {
                            for ( uint64_t e = 0; e < elements; e += 2 )
                            {
                                target.set32( e / 2, nref.get32( e ) + nref.get32( e + 1 ) );
                                target.set32( ( elements + e ) / 2, mref.get32( e ) + mref.get32( e + 1 ) );
                            }
                        }
                        else if ( 8 == ebytes )
                        {
                            target.set64( 0, nref.get64( 0 ) + nref.get64( 1 ) );
                            target.set64( 1, mref.get64( 0 ) + mref.get64( 1 ) );
                        }
                        vregs[ d ] = target;
                    }
                    else if ( ( 0x4e == hi8 || 0x0e == hi8 ) && bit15 && 0 == bits14_11 && bit10 ) // ADD <Vd>.<T>, <Vn>.<T>, <Vm>.<T>.   add vector
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t m = opbits( 16, 5 );
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        vec16_t target;
                        //tracer.Trace( "elements: %llu, ebytes %llu, size %llu, esize %llu\n", elements, ebytes, size, esize );

                        vec16_t & vn = vregs[ n ];
                        vec16_t & vm = vregs[ m ];

                        if ( 1 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set8( e, vn.get8( e ) + vm.get8( e ) );
                        else if ( 2 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set16( e, vn.get16( e ) + vm.get16( e ) );
                        else if ( 4 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set32( e, vn.get32( e ) + vm.get32( e ) );
                        else if ( 8 == ebytes )
                            for ( uint64_t e = 0; e < elements; e++ )
                                target.set64( e, vn.get64( e ) + vm.get64( e ) );
                        else
                            unhandled();

                        vregs[ d ] = target;
                    }
                    else if ( 0xb == bits14_11 && 0 == bits20_16 && !bit15 ) // CNT <Vd>.<T>, <Vn>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        if ( 0 != size )
                            unhandled();

                        uint64_t elements = ( 0 == Q ) ? 1 : 2;
                        uint64_t bitcount = 0;
                        for ( uint64_t x = 0; x < elements; x++ )
                            bitcount += count_bits( vregs[ n ].get64( x ) );
                        vregs[ d ].set64( 0, bitcount );
                        vregs[ d ].set64( 1, 0 );
                    }
                    else if ( ( 0x4e == hi8 || 0x0e == hi8 ) && 0x11 == bits20_16 && bit15 && 7 == bits14_11 ) // ADDV <V><d>, <Vn>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        if ( 3 == size )
                            unhandled();

                        // even though arm64 doc says types can include S, always use integer math
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = ( Q ? 16 : 8 ) / ebytes;

                        if ( 1 == ebytes )
                        {
                            uint8_t total = 0;
                            for ( uint64_t x = 0; x < elements; x++ )
                                total += vregs[ n ].get8( x );
                            zero_vreg( d );
                            vregs[ d ].set8( 0, total );
                        }
                        else if ( 2 == ebytes )
                        {
                            uint16_t total = 0;
                            for ( uint64_t x = 0; x < elements; x++ )
                                total += vregs[ n ].get16( x );
                            zero_vreg( d );
                            vregs[ d ].set16( 0, total );
                        }
                        else if ( 4 == ebytes )
                        {
                            uint32_t total = 0;
                            for ( uint64_t x = 0; x < elements; x++ )
                                total += vregs[ n ].get32( x );
                            zero_vreg( d );
                            vregs[ d ].set32( 0, total );
                        }
                        else
                            unhandled();
                    }
                    else if ( 1 == bits20_16 && !bit15 && 5 == bits14_11 && !bit10 ) // xtn, xtn2 XTN{2} <Vd>.<Tb>, <Vn>.<Ta>
                    {
                        uint64_t size = opbits( 22, 2 );
                        if ( 3 == size )
                            unhandled();

                        uint64_t target_esize = 8ull << size;
                        uint64_t source_ebytes = target_esize * 2 / 8;
                        uint64_t target_ebytes = target_esize / 8;
                        uint64_t elements = 64 / target_esize;
                        uint64_t result = 0;
                        uint8_t * psrc = vreg_ptr( n, 0 );
                        //tracer.Trace( "  xtn. Q %llu, elements %llu, target_esize %llu, size %llu\n", Q, elements, target_esize, size );
                        assert( target_ebytes <= sizeof( result ) );

                        for ( uint64_t x = 0; x < elements; x++ )
                        {
                            assert( ( ( x * target_ebytes ) + target_ebytes ) <= sizeof( result ) );
                            mcpy( ( (uint8_t *) &result ) + x * target_ebytes, psrc + x * source_ebytes, target_ebytes );
                        }

                        result = consider_endian64( result );

                        if ( Q )
                            vregs[ d ].set64( 1, result ); // don't modifiy the lower half
                        else
                        {
                            vregs[ d ].set64( 0, result );
                            vregs[ d ].set64( 1, 0 ); // zero the upper half
                        }
                    }
                    else
                        unhandled();
                }
                else // !bit21
                {
                    if ( !bit15 && ( 0x1e == bits14_10 || 0xe == bits14_10 ) ) // ZIP1/2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        uint64_t part = opbit( 14 );
                        uint64_t pairs = elements / 2;
                        uint64_t base_amount = part * pairs;
                        vec16_t target;
                        uint8_t * ptarget = (uint8_t *) &target;

                        for ( uint64_t p = 0; p < pairs; p++ )
                        {
                            assert( ( ( ( 2 * p ) * ebytes ) + ebytes ) <= sizeof( target ) );
                            mcpy( ptarget + 2 * p * ebytes, vreg_ptr( n, ( base_amount + p ) * ebytes ), ebytes );
                            assert( ( ( ( 2 * p + 1 ) * ebytes ) + ebytes ) <= sizeof( target ) );
                            mcpy( ptarget + ( 2 * p + 1 ) * ebytes, vreg_ptr( m, ( base_amount + p ) * ebytes ), ebytes );
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits23_21 && !bit15 && 0 == bits12_10 ) // TBL <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B, <Vn+2>.16B, <Vn+3>.16B },
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t len = opbits( 13, 2 );
                        uint64_t elements = datasize / 8;
                        uint64_t reg_count = len + 1;
                        vec16_t src[ 4 ];
                        assert( reg_count <= _countof( src ) );
                        for ( uint64_t i = 0; i < reg_count; i++ )
                            src[ i ] = vregs[ ( n + i ) % 32 ];
                        vec16_t target;

                        for ( uint64_t i = 0; i < elements; i++ )
                        {
                            uint64_t index = vregs[ m ].get8( i );
                            if ( index < ( 16 * reg_count ) )
                            {
                                uint64_t src_item = index / 16;
                                uint64_t src_index = index % 16;
                                assert( src[ 0 ].get8( index ) == src[ src_item ].get8( src_index ) ); // index will reach into subsequent src entries!
                                target.set8( i, src[ src_item ].get8( src_index ) );
                            }
                        }
                        vregs[ d ] = target;
                    }
                    else if ( !bit15 && ( 0xd == bits14_11 || 5 == bits14_11 ) && !bit10 ) // TRN1/2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t m = opbits( 16, 5 );
                        uint64_t size = opbits( 22, 2 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        uint64_t pairs = elements / 2;
                        uint64_t part = opbit( 14 ); // TRN1 vs TRN2
                        vec16_t target;
                        uint8_t * ptarget = (uint8_t *) &target;

                        for ( uint64_t p = 0; p < pairs; p++ )
                        {
                            assert( ( ( ( 2 * p ) * ebytes ) + ebytes ) <= sizeof( target ) );
                            mcpy( ptarget + ( ( 2 * p ) * ebytes ), vreg_ptr( n, ( 2 * p + part ) * ebytes ), ebytes );
                            assert( ( ( ( 2 * p + 1 ) * ebytes ) + ebytes ) <= sizeof( target ) );
                            mcpy( ptarget + ( ( 2 * p + 1 ) * ebytes ), vreg_ptr( m, ( 2 * p + part ) * ebytes ), ebytes );
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0x4e == hi8 && 0 == bits23_21 && !bit15 && 3 == bits14_11 && bit10 ) // INS <Vd>.<Ts>[<index>], <R><n>
                    {
                        uint64_t index = 0;
                        if ( imm5 & 1 )
                        {
                            index = get_bits( imm5, 1, 4 );
                            vregs[ d ].set8( index, (uint8_t) regs[ n ] );
                        }
                        else if ( imm5 & 2 )
                        {
                            index = get_bits( imm5, 2, 3 );
                            vregs[ d ].set16( index, (uint16_t) regs[ n ] );
                        }
                        else if ( imm5 & 4 )
                        {
                            index = get_bits( imm5, 3, 2 );
                            vregs[ d ].set32( index, (uint32_t) regs[ n ] );
                        }
                        else if ( imm5 & 8 )
                        {
                            index = get_bit( imm5, 4 );
                            vregs[ d ].set64( index, (uint64_t) regs[ n ] );
                        }
                        else
                            unhandled();
                    }
                    else if ( !bit15 && ( 7 == bits14_11 || 5 == bits14_11 ) && bit10 )
                    {
                        // UMOV <Wd>, <Vn>.<Ts>[<index>]    ;    UMOV <Xd>, <Vn>.D[<index>]    ;     SMOV <Wd>, <Vn>.<Ts>[<index>]    ;
                        uint64_t size = lowest_set_bit_nz( imm5 & ( ( 7 == bits14_11 ) ? 0xf : 7 ) );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        datasize = 32ull << Q;
                        uint64_t bits_to_copy = 4 - size;
                        uint64_t index = get_bits( imm5, 4 + 1 - bits_to_copy, bits_to_copy );
                        // tracer.Trace( "mov, size %llu, esize %llu, ebytes %llu, datasize %llu, index %llu\n", size, esize, ebytes, datasize, index );

                        uint64_t val = 0;
                        if ( 1 == ebytes )
                            val = vregs[ n ].get8( index );
                        else if ( 2 == ebytes )
                            val = vregs[ n ].get16( index );
                        else if ( 4 == ebytes )
                            val = vregs[ n ].get32( index );
                        else
                            val = vregs[ n ].get64( index );

                        if ( 5 == bits14_11 )
                            val = sign_extend( val, esize - 1 );
                        if ( 31 != d )
                            regs[ d ] = Q ? val : (uint32_t) val;
                    }
                    else if ( !bit15 && ( 0x3 == bits14_11 || 0xb == bits14_11 ) && !bit10 ) // UZP2 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>    ;    UZP1 <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
                    {
                        uint64_t size = opbits( 22, 2 );
                        uint64_t m = imm5;
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        uint64_t part = opbit( 14 ); // UZP2 is 1, UZP1 is 0
                        vec16_t target;
                        uint8_t * ptarget = (uint8_t *) &target;
                        uint64_t second_offset = elements / 2 * ebytes;
                        //tracer.Trace( "elements %llu, ebytes %llu, second_offset %llu\n", elements, ebytes, second_offset );
                        for ( uint64_t e = 0; e < elements / 2; e++ )
                        {
                            assert( ( ( e * ebytes ) + ebytes ) <= sizeof( target ) );
                            mcpy( ptarget + e * ebytes, vreg_ptr( n, ( e * 2 + part ) * ebytes ), ebytes ); // odd or even from n into lower half of d
                            assert( ( second_offset + ( e * ebytes ) + ebytes ) <= sizeof( target ) );
                            mcpy( ptarget + second_offset + e * ebytes, vreg_ptr( m, ( e * 2 + part ) * ebytes ), ebytes ); // odd or even from m into upper half of d
                        }
                        vregs[ d ] = target;
                    }
                    else if ( 0 == bits23_21 && !bit15 && 1 == bits14_11 && bit10 ) // DUP <Vd>.<T>, <R><n>
                    {
                        uint64_t size = lowest_set_bit_nz( imm5 & 0xf );
                        uint64_t esize = 8ull << size;
                        uint64_t elements = datasize / esize;
                        uint64_t val = val_reg_or_zr( n );
                        uint8_t * pmem = vreg_ptr( d, 0 );
                        uint64_t ebytes = esize / 8;
                        memset( pmem, 0, sizeof( vregs[ d ] ) );
                        val = consider_endian64( val );

                        for ( uint64_t e = 0; e < elements; e++ )
                            mcpy( pmem + ( e * ebytes ), &val, ebytes );
                        //tracer.TraceBinaryData( & vregs[ d ], sizeof( vregs[ d ] ), 4 );
                    }
                    else if ( 0 == bits23_21 && !bit15 && 0 == bits14_11 && bit10 ) // DUP <Vd>.<T>, <Vn>.<Ts>[<index>]
                    {
                        uint64_t size = lowest_set_bit_nz( imm5 & 0xf );
                        uint64_t index = get_bits( imm5, size + 1, 4 - ( size + 1 ) + 1 );
                        uint64_t esize = 8ull << size;
                        uint64_t ebytes = esize / 8;
                        uint64_t elements = datasize / esize;
                        uint64_t element = 0;
                        //tracer.Trace( "index %llu, indbytes %llu, ebytes: %llu, elements %llu\n", index, indbytes, ebytes, elements );
                        mcpy( &element, vreg_ptr( n, index * ebytes ), ebytes );
                        for ( uint64_t e = 0; e < elements; e++ )
                            mcpy( vreg_ptr( d, e * ebytes ), &element, ebytes );
                    }
                    else
                        unhandled();
                }

                trace_vregs();
                break;
            }
            case 0x1e: // FMOV <Wd>, <Hn>    ;    FMUL                ;    FMOV <Wd>, imm       ;    FCVTZU <Wd>, <Dn>    ;    FRINTA <Dd>, <Dn>    ;    FMAXNM <Dd>, <Dn>, <Dm>
                       // FMAX <Dd>, <Dn>, <Dm> ; FMINNM <Dd>, <Dn>, <Dm>  ; FMIN <Dd>, <Dn>, <Dm> ; FRINTZ <Dd>, <Dn>    ;    FRINTP <Dd>, <Dn>
            case 0x9e: // FMOV <Xd>, <Hn>    ;    UCVTF <Hd>, <Dn>    ;    FCVTZU <Xd>, <Dn>    ;    FCVTAS <Xd>, <Dn>    ;    FCVTMU <Xd>, <Dn>
            {
                uint64_t sf = opbit( 31 );
                uint64_t ftype = opbits( 22, 2 );
                uint64_t bit21 = opbit( 21 );
                uint64_t bit11 = opbit( 11 );
                uint64_t bit10 = opbit( 10 );
                uint64_t bit4 = opbit( 4 );
                uint64_t bits21_19 = opbits( 19, 3 );
                uint64_t rmode = opbits( 19, 2 );
                uint64_t bits18_16 = opbits( 16, 3 );
                uint64_t bits18_10 = opbits( 10, 9 );
                uint64_t bits15_10 = opbits( 10, 6 );
                uint64_t n = opbits( 5, 5 );
                uint64_t d = opbits( 0, 5 );

                if ( 0x1e == hi8 && 4 == bits21_19 && 0x130 == bits18_10 ) // FRINTP <Dd>, <Dn>
                {
                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, (float) round_double( vregs[ n ].getf( 0 ), FPRounding_POSINF ) );
                    else if ( 1 == ftype )
                        vregs[ d ].setd( 0, round_double( vregs[ n ].getd( 0 ), FPRounding_POSINF ) );
                    else
                        unhandled();
                }
                else if ( 0x1e == hi8 && 4 == bits21_19 && 0x170 == bits18_10 ) // FRINTZ <Dd>, <Dn>
                {
                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, (float) round_double( vregs[ n ].getf( 0 ), FPRounding_ZERO ) );
                    else if ( 1 == ftype )
                        vregs[ d ].setd( 0, round_double( vregs[ n ].getd( 0 ), FPRounding_ZERO ) );
                    else
                        unhandled();
                }
                else if ( 0x1e == hi8 && bit21 && ( 0x12 == bits15_10 || 0x1a == bits15_10 ) ) // FMAX <Dd>, <Dn>, <Dm>    ;    FMAXNM <Dd>, <Dn>, <Dm>
                {
                    uint64_t m = opbits( 16, 5 );
                    bool isFMAX = ( 0x12 == bits15_10 );
                    bool fpcr_ah = get_bit( fpcr, 1 );
                    bool useSecond = isFMAX && fpcr_ah;
                    double nval = 0.0, mval = 0.0, result = 0.0;

                    if ( 0 == ftype )
                    {
                        nval = vregs[ n ].getf( 0 );
                        mval = vregs[ m ].getf( 0 );
                    }
                    else if ( 1 == ftype )
                    {
                        nval = vregs[ n ].getd( 0 );
                        mval = vregs[ m ].getd( 0 );
                    }
                    else
                        unhandled();

                    if ( useSecond && ( ( 0.0 == nval && 0.0 == mval ) || isnan( nval ) || isnan( mval ) ) )
                        result = mval;
                    else
                        result = do_fmax( nval, mval );

                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, (float) result );
                    else
                        vregs[ d ].setd( 0, result );

                    trace_vregs();
                }
                else if ( 0x1e == hi8 && bit21 && ( 0x16 == bits15_10 || 0x1e == bits15_10 ) ) // FMIN <Dd>, <Dn>, <Dm>    ;    FMINNM <Dd>, <Dn>, <Dm>
                {
                    uint64_t m = opbits( 16, 5 );
                    bool isFMIN = ( 0x16 == bits15_10 );
                    bool fpcr_ah = get_bit( fpcr, 1 );
                    bool useSecond = isFMIN && fpcr_ah;
                    double nval = 0.0, mval = 0.0, result = 0.0;

                    if ( 0 == ftype )
                    {
                        nval = vregs[ n ].getf( 0 );
                        mval = vregs[ m ].getf( 0 );
                    }
                    else if ( 1 == ftype )
                    {
                        nval = vregs[ n ].getd( 0 );
                        mval = vregs[ m ].getd( 0 );
                    }
                    else
                        unhandled();

                    if ( useSecond && ( ( 0.0 == nval && 0.0 == mval ) || isnan( nval ) || isnan( mval ) ) )
                        result = mval;
                    else
                        result = do_fmin( nval, mval );

                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, (float) result );
                    else
                        vregs[ d ].setd( 0, result );

                    trace_vregs();
                }
                else if ( 0x1e == hi8 && 4 == bits21_19 && 0x150 == bits18_10 ) // FRINTM <Dd>, <Dn>
                {
                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, (float) round_double( (double) vregs[ n ].getf( 0 ), FPRounding_NEGINF ) );
                    else if ( 1 == ftype )
                        vregs[ d ].setd( 0, round_double( vregs[ n ].getd( 0 ), FPRounding_NEGINF ) );
                    else
                        unhandled();
                }
                else if ( 0x1e == hi8 && bit21 && !bit11 && bit10 && bit4 ) // FCCMPE <Sn>, <Sm>, #<nzcv>, <cond>    ;    FCCMPE <Dn>, <Dm>, #<nzcv>, <cond>
                {
                    uint64_t m = opbits( 16, 5 );
                    uint64_t cond = opbits( 12, 4 );

                    if ( check_conditional( cond ) )
                    {
                        //tracer.Trace( "condition holds, so doing compare\n" );
                        double result = 0.0;
                        if ( 0 == ftype )
                            result = do_fsub( vregs[ n ].getf( 0 ), vregs[ m ].getf( 0 ) );
                        else if ( 1 == ftype )
                            result = do_fsub( vregs[ n ].getd( 0 ), vregs[ m ].getd( 0 ) );
                        else
                            unhandled();

                        set_flags_from_double( result );
                    }
                    else
                    {
                        uint64_t nzcv = opbits( 0, 4 );
                        set_flags_from_nzcv( nzcv );
                    }
                }
                else if ( 3 == bits21_19 && 0 == bits18_16 ) // FCVTZS <Xd>, <Dn>, #<fbits>
                {
                    if ( 31 == d )
                        break;
                    uint64_t scale = opbits( 10, 6 );
                    uint64_t fracbits = 64 - scale;
                    double src = 0.0;
                    if ( 0 == ftype )
                        src = vregs[ n ].getf( 0 );
                    else if ( 1 == ftype )
                        src = vregs[ n ].getd( 0 );
                    else
                        unhandled();

                    uint64_t result = 0;

                    if ( sf )
                        result = double_to_fixed_int64( src, fracbits, FPRounding_ZERO );
                    else
                        result = (uint32_t) double_to_fixed_int32( src, fracbits, FPRounding_ZERO );

                    regs[ d ] = result;
                }
                else if ( 6 == bits21_19 && 0x40 == bits18_10 ) // FCVTMU <Xd>, <Dn>
                {
                    if ( !sf && 0 == ftype )
                        regs[ d ] = double_to_fixed_uint32( (double) vregs[ n ].getf( 0 ), 0, FPRounding_NEGINF );
                    else if ( sf && 0 == ftype )
                        regs[ d ] = double_to_fixed_uint64( (double) vregs[ n ].getf( 0 ), 0, FPRounding_NEGINF );
                    else if ( !sf && 1 == ftype )
                        regs[ d ] = double_to_fixed_uint32( vregs[ n ].getd( 0 ), 0, FPRounding_NEGINF );
                    else if ( sf && 1 == ftype )
                        regs[ d ] = double_to_fixed_uint64( vregs[ n ].getd( 0 ), 0, FPRounding_NEGINF );
                    else
                        unhandled();
                }
                else if ( 4 == bits21_19 && 0x100 == bits18_10 ) // FCVTAS <Xd>, <Dn>
                {
                    if ( !sf && 0 == ftype )
                        regs[ d ] = (uint32_t) (int32_t) round( vregs[ n ].getf( 0 ) );
                    else if ( sf && 0 == ftype )
                        regs[ d ] = (uint64_t) (int64_t) (int32_t) round( vregs[ n ].getf( 0 ) );
                    else if ( !sf && 1 == ftype )
                        regs[ d ] = (uint32_t) (int32_t) round( vregs[ n ].getd( 0 ) );
                    else if ( sf && 1 == ftype )
                        regs[ d ] = (uint64_t) (int64_t) (int32_t) round( vregs[ n ].getd( 0 ) );
                    else
                        unhandled();
                }
                else if ( 0x1e == hi8 && 4 == bits21_19 && 0x190 == bits18_10 ) // FRINTA <Dd>, <Dn>
                {
                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, (float) round_double( (double) vregs[ n ].getf( 0 ), FPRounding_TIEAWAY ) );
                    else if ( 1 == ftype )
                        vregs[ d ].setd( 0, round_double( vregs[ n ].getd( 0 ), FPRounding_TIEAWAY ) );
                    else
                        unhandled();
                    trace_vregs();
                }
                else if ( ( 0x180 == ( bits18_10 & 0x1bf ) ) && ( bit21 ) && ( 0 == ( rmode & 2 ) ) ) // fmov reg, vreg  OR  fmov vreg, reg
                {
                    uint64_t opcode = opbits( 16, 3 );
                    uint64_t nval = val_reg_or_zr( n );
                    if ( 0 == sf )
                    {
                        if ( 0 != rmode )
                            unhandled();

                        if ( 3 == ftype )
                        {
                            if (  6 == opcode )
                                regs[ d ] = vregs[ n ].get16( 0 );
                            else if ( 7 == opcode )
                            {
                                zero_vreg( d );
                                vregs[ d ].set16( 0, (uint16_t) nval );
                            }
                            else
                                unhandled();
                        }
                        else if ( 0 == ftype )
                        {
                            if ( 7 == opcode )
                            {
                                zero_vreg( d );
                                vregs[ d ].set32( 0, (uint32_t) nval );
                            }
                            else if ( 6 == opcode )
                                regs[ d ] = vregs[ n ].get32( 0 );
                            else
                                unhandled();
                        }
                        else
                            unhandled();

                        trace_vregs();
                    }
                    else
                    {
                        if ( 0 == rmode )
                        {
                            if ( 3 == ftype && 6 == opcode )
                                regs[ d ] = vregs[ n ].get16( 0 );
                            else if ( 3 == ftype && 7 == opcode )
                            {
                                zero_vreg( d );
                                vregs[ d ].set16( 0, nval & 0xffff );
                            }
                            else if ( 1 == ftype && 7 == opcode )
                            {
                                vregs[ d ].set64( 0, nval );
                                vregs[ d ].set64( 1, 0 );
                            }
                            else if ( 1 == ftype && 6 == opcode )
                                regs[ d ] = vregs[ n ].get64( 0 );
                            else
                                unhandled();

                            trace_vregs();
                        }
                        else
                        {
                            if ( 2 == ftype && 7 == opcode )
                                vregs[ d ].set64( 1, nval );
                            else if ( 2 == ftype && 6 == opcode )
                                regs[ d ] = vregs[ n ].get64( 1 );
                            else
                                unhandled();
                        }
                    }
                }
                else if ( 0x40 == bits18_10 && bit21 && 3 == rmode ) // FCVTZU <Wd>, <Dn>
                {
                    if ( 31 == d )
                        break;

                    double src = 0.0;
                    if ( 0 == ftype )
                        src = vregs[ n ].getf( 0 );
                    else if ( 1 == ftype )
                        src = vregs[ n ].getd( 0 );
                    else
                        unhandled();

                    uint64_t result = 0;
                    if ( src > 0.0 )
                    {
                        if ( sf )
                        {
                            if ( src > (double) UINT64_MAX )
                                result = UINT64_MAX;
                            else
                                result = (uint64_t) src;
                        }
                        else
                        {
                            if ( src > (double) UINT32_MAX )
                                result = UINT32_MAX;
                            else
                                result = (uint32_t) src;
                        }
                    }
                    regs[ d ] = result;
                }
                else if ( ( 0x40 == ( bits18_10 & 0x1c0 ) ) && !bit21 && 3 == rmode ) // FCVTZU <Wd>, <Dn>, #<fbits>
                {
                    double src = 0.0;
                    if ( 31 == d )
                        break;

                    if ( 0 == ftype )
                        src = (double) vregs[ n ].getf( 0 );
                    else if ( 1 == ftype )
                        src = vregs[ n ].getd( 0 );
                    else
                        unhandled();

                    uint64_t result = 0;

                    if ( src > 0.0 )
                    {
                        uint64_t scale = opbits( 10, 6 );
                        uint64_t fracbits = 64 - scale;

                        if ( sf )
                        {
                            if ( src > (double) UINT64_MAX )
                                result = UINT64_MAX;
                            else
                                result = double_to_fixed_uint64( src, fracbits, FPRounding_ZERO );
                        }
                        else
                        {
                            if ( src > (double) UINT32_MAX )
                                result = UINT32_MAX;
                            else
                                result = double_to_fixed_uint32( src, fracbits, FPRounding_ZERO );
                        }
                    }
                    regs[ d ] = result;
                }
                else if ( ( 0x1e == hi8 ) && ( 4 == ( bits18_10 & 7 ) ) && ( bit21 ) ) // fmov scalar immediate
                {
                    uint64_t fltsize = ( 2 == ftype ) ? 64 : ( 8ull << ( ftype ^ 2 ) );
                    assert( fltsize <= 64 );
                    uint64_t imm8 = opbits( 13, 8 );
                    uint64_t val = vfp_expand_imm( imm8, fltsize );
                    zero_vreg( d );
                    if ( 64 == fltsize )
                        vregs[ d ].set64( 0, val );
                    else if ( 32 == fltsize )
                        vregs[ d ].set32( 0, (uint32_t) val );
                    else
                        unhandled();
                    trace_vregs();
                }
                else if ( ( 0x1e == hi8 ) && ( 2 == ( bits18_10 & 0x3f ) ) && ( bit21 ) ) // fmul (scalar)
                {
                    uint64_t m = opbits( 16, 5 );
                    if ( 0 == ftype ) // single-precision
                    {
                        vregs[ d ].setf( 0, (float) do_fmul( vregs[ n ].getf( 0 ), vregs[ m ].getf( 0 ) ) );
                        memset( vreg_ptr( d, 4 ), 0, 12 );
                    }
                    else if ( 1 == ftype ) // double-precision
                    {
                        vregs[ d ].setd( 0, do_fmul( vregs[ n ].getd( 0 ), vregs[ m ].getd( 0 ) ) );
                        memset( vreg_ptr( d, 8 ), 0, 8 );
                    }
                    else
                        unhandled();
                    trace_vregs();
                }
                else if ( ( 0x1e == hi8 ) && ( 0x90 == ( bits18_10 & 0x19f ) ) && ( bit21 ) ) // fcvt
                {
                    uint64_t opc = opbits( 15, 2 );
                    if ( 0 == ftype )
                    {
                        if ( 1 == opc ) // single to double
                        {
                            vregs[ d ].setd( 0, (double) vregs[ n ].getf( 0 ) );
                            memset( vreg_ptr( d, 8 ), 0, 8 );
                        }
                        else
                            unhandled();
                    }
                    else if ( 1 == ftype )
                    {
                        if ( 0 == opc ) // double to single
                        {
                            vregs[ d ].setf( 0, (float) vregs[ n ].getd( 0 ) );
                            memset( vreg_ptr( d, 4 ), 0, 12 );
                        }
                        else
                            unhandled();
                    }
                    else
                        unhandled();

                    trace_vregs();
                }
                else if ( ( 0x1e == hi8 ) && ( 0x10 == bits18_10 ) && ( 4 == bits21_19 ) ) // fmov
                    vregs[ d ] = vregs[ n ];
                else if ( ( 0x1e == hi8 ) && ( 8 == ( bits18_10 & 0x3f ) ) && ( bit21 ) ) // fcmp and fcmpe (no signaling yet)
                {
                    uint64_t m = opbits( 16, 5 );
                    uint64_t opc = opbits( 3, 2 );
                    double result = 0.0;

                    if ( 3 == ftype && ( ( 0 == opc ) || ( 2 == opc ) ) )
                        unhandled(); // Hn, Hm
                    else if ( 3 == ftype && ( 0 == m && ( ( 1 == opc ) || 3 == opc ) ) )
                        unhandled(); // Hn, 0.0
                    else if ( 0 == ftype && ( ( 0 == opc ) || ( 2 == opc ) ) )
                    {
                        if ( isinf( vregs[ n ].getf( 0 ) ) && isinf( vregs[ m ].getf( 0 ) ) )
                        {
                            bool signn = signbit( vregs[ n ].getf( 0 ) );
                            bool signm = signbit( vregs[ m ].getf( 0 ) );
                            if ( signn && !signm )
                                result = -1.0;
                            else if ( !signn && signm )
                                result = 1.0;
                            else
                                result = 0.0;
                        }
                        else if ( isnan( vregs[ n ].getf( 0 ) ) || isnan( vregs[ m ].getf( 0 ) ) )
                            result = MY_NAN;
                        else
                            result = vregs[ n ].getf( 0 ) - vregs[ m ].getf( 0 );
                    }
                    else if ( 0 == ftype && 0 == m && ( ( 1 == opc ) || ( 3 == opc ) ) )
                        result = vregs[ n ].getf( 0 ); // - 0.0f;
                    else if ( 1 == ftype && ( ( 0 == opc ) || ( 2 == opc ) ) )
                    {
                        if ( isinf( vregs[ n ].getd( 0 ) ) && isinf( vregs[ m ].getd( 0 ) ) )
                        {
                            bool signn = signbit( vregs[ n ].getd( 0 ) );
                            bool signm = signbit( vregs[ m ].getd( 0 ) );
                            if ( signn && !signm )
                                result = -1.0;
                            else if ( !signn && signm )
                                result = 1.0;
                            else
                                result = 0.0;
                        }
                        else if ( isnan( vregs[ n ].getd( 0 ) ) || isnan( vregs[ m ].getd( 0 ) ) )
                            result = MY_NAN;
                        else
                            result = vregs[ n ].getd( 0 ) - vregs[ m ].getd( 0 );
                    }
                    else if ( 1 == ftype && 0 == m && ( ( 1 == opc ) || ( 3 == opc ) ) )
                        result = vregs[ n ].getd( 0 ); // - 0.0;
                    else
                        unhandled();

                    set_flags_from_double( result );
                }
                else if ( ( 0x1e == hi8 ) && ( 0x30 == bits18_10 ) && ( 4 == bits21_19 ) ) // fabs (scalar)
                {
                    if ( 3 == ftype )
                        unhandled();
                    else if ( 0 == ftype )
                    {
                        vregs[ d ].setf( 0, fabsf( vregs[ n ].getf( 0 ) ) );
                        memset( vreg_ptr( d, 4 ), 0, 12 );
                    }
                    else if ( 1 == ftype )
                    {
                        vregs[ d ].setd( 0, fabs( vregs[ n ].getd( 0 ) ) );
                        memset( vreg_ptr( d, 8 ), 0, 8 );
                    }
                    else
                        unhandled();
                    trace_vregs();
                }
                else if ( 0x1e == hi8 && ( 6 == ( 0x3f & bits18_10 ) ) && bit21 ) // fdiv v, v, v
                {
                    uint64_t m = opbits( 16, 5 );
                    if ( 0 == ftype ) // single-precision
                        vregs[ d ].setf( 0, (float) do_fdiv( vregs[ n ].getf( 0 ), vregs[ m ].getf( 0 ) ) );
                    else if ( 1 == ftype ) // double-precision
                        vregs[ d ].setd( 0, do_fdiv( vregs[ n ].getd( 0 ), vregs[ m ].getd( 0 ) ) );
                    else
                        unhandled();
                    trace_vregs();
                }
                else if ( 0x1e == hi8 && ( 0xa == ( 0x3f & bits18_10 ) ) && bit21 ) // fadd v, v, v
                {
                    uint64_t m = opbits( 16, 5 );
                    if ( 0 == ftype ) // single-precision
                        vregs[ d ].setf( 0, (float) do_fadd( vregs[ n ].getf( 0 ), vregs[ m ].getf( 0 ) ) );
                    else if ( 1 == ftype ) // double-precision
                        vregs[ d ].setd( 0, do_fadd( vregs[ n ].getd( 0 ), vregs[ m ].getd( 0 ) ) );
                    else
                        unhandled();
                    trace_vregs();
                }
                else if ( 0x1e == hi8 && ( 0xe == ( 0x3f & bits18_10 ) ) && bit21 ) // fsub v, v, v
                {
                    uint64_t m = opbits( 16, 5 );
                    if ( 0 == ftype ) // single-precision
                        vregs[ d ].setf( 0, (float) do_fsub( vregs[ n ].getf( 0 ), vregs[ m ].getf( 0 ) ) );
                    else if ( 1 == ftype ) // double-precision
                        vregs[ d ].setd( 0, do_fsub( vregs[ n ].getd( 0 ), vregs[ m ].getd( 0 ) ) );
                    else
                        unhandled();
                }
                else if ( 0x80 == bits18_10 && bit21 && 0 == rmode ) // SCVTF (scalar, integer)
                {
                    uint64_t nval = val_reg_or_zr( n );
                    if ( !sf )
                        nval = sign_extend( (uint32_t) nval, 31 );

                    zero_vreg( d );
                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, (float) (int64_t) nval );
                    else if ( 1 == ftype )
                        vregs[ d ].setd( 0, (double) (int64_t) nval );
                    else
                        unhandled();
                }
                else if ( 0x70 == bits18_10 && bit21 && 0 == rmode ) // fsqrt s#, s#
                {
                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, sqrtf( vregs[ n ].getf( 0 ) ) );
                    else if ( 1 == ftype )
                        vregs[ d ].setd( 0, sqrt( vregs[ n ].getd( 0 ) ) );
                    else
                        unhandled();
                }
                else if ( bit21 && ( 3 == ( 3 & bits18_10 ) ) ) // fcsel
                {
                    uint64_t m = opbits( 16, 5 );
                    uint64_t cond = opbits( 12, 4 );
                    bool met = check_conditional( cond );

                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, met ? vregs[ n ].getf( 0 ) : vregs[ m ].getf( 0 ) );
                    else if ( 1 == ftype )
                        vregs[ d ].setd( 0, met ? vregs[ n ].getd( 0 ) : vregs[ m ].getd( 0 ) );
                    else
                        unhandled();
                }
                else if ( bit21 && ( 0x50 == bits18_10 ) ) // fneg (scalar)
                {
                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, - vregs[ n ].getf( 0 ) );
                    else if ( 1 == ftype )
                        vregs[ d ].setd( 0, - vregs[ n ].getd( 0 ) );
                    else
                        unhandled();
                }
                else if ( bit21 && 0 == bits18_10 && 3 == rmode ) // FCVTZS <Wd>, <Dn>
                {
                    if ( 0 == ftype )
                    {
                        if ( sf )
                            regs[ d ] = double_to_fixed_int64( (double) vregs[ n ].getf( 0 ), 0, FPRounding_ZERO );
                        else
                            regs[ d ] = (uint32_t) double_to_fixed_int32( (double) vregs[ n ].getf( 0 ), 0, FPRounding_ZERO );
                    }
                    else if ( 1 == ftype )
                    {
                        if ( sf )
                            regs[ d ] = double_to_fixed_int64( vregs[ n ].getd( 0 ), 0, FPRounding_ZERO );
                        else
                            regs[ d ] = (uint32_t) double_to_fixed_int32( vregs[ n ].getd( 0 ), 0, FPRounding_ZERO );
                    }
                    else
                        unhandled();
                }
                else if ( bit21 && ( 1 == ( bits18_10 & 3 ) ) && ( 0 == opbit( 4 ) ) ) // fccmp
                {
                    uint64_t m = opbits( 16, 5 );
                    uint64_t cond = opbits( 12, 4 );

                    if ( check_conditional( cond ) )
                    {
                        double result = 0.0;
                        double nval, mval;
                        if ( 0 == ftype )
                        {
                            nval = vregs[ n ].getf( 0 );
                            mval = vregs[ m ].getf( 0 );
                        }
                        else
                        {
                            nval = vregs[ n ].getd( 0 );
                            mval = vregs[ m ].getd( 0 );
                        }

                        if ( isinf( nval ) && isinf( mval ) )
                            result = 0.0;
                        else if ( isnan( nval ) || isnan( mval ) )
                            result = MY_NAN;
                        else
                            result = do_fsub( vregs[ n ].getd( 0 ), vregs[ m ].getd( 0 ) );

                        set_flags_from_double( result );
                    }
                    else
                    {
                        uint64_t nzcv = opbits( 0, 4 );
                        set_flags_from_nzcv( nzcv );
                    }
                }
                else if ( bit21 && ( 0xc0 == ( 0x1c0 & bits18_10 ) ) && 0 == rmode ) // UCVTF <Hd>, <Wn>, #<fbits>
                {
                    uint64_t val = val_reg_or_zr( n );
                    if ( 0 == sf )
                        val = (uint32_t) val;

                    zero_vreg( d );

                    if ( 0 == ftype )
                        vregs[ d ].setf( 0, (float) val );
                    else if ( 1 == ftype )
                        vregs[ d ].setd( 0, (double) val );
                    else
                        unhandled();
                }
                else
                    unhandled();
                break;
            }
            case 0x0c:
            case 0x4c: // LD1 { <Vt>.<T> }, [<Xn|SP>]    ;    LD2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
                       // ST2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]    ;    ST2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <imm>    ;    ST2 { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <Xm>
                       // LD3 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>]
                       // LD3 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <imm>
                       // LD3 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <Xm>
                       // LD4 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>]
                       // LD4 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <imm>
                       // LD4 { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <Xm>
            {
                uint64_t Q = opbit( 30 );
                uint64_t L = opbit( 22 ); // load vs. store
                uint64_t post_index = opbit( 23 );
                uint64_t opcode = opbits( 12, 4 );
                uint64_t size = opbits( 10, 2 );
                uint64_t bits23_21 = opbits( 21, 3 );
                uint64_t m = opbits( 16, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t t = opbits( 0, 5 );

                if ( 2 != bits23_21 && 6 != bits23_21 && 0 != bits23_21 && 4 != bits23_21)
                    unhandled();

                if ( ( 2 & opcode ) || 8 == opcode || 4 == opcode || 0 == opcode ) // LD1 / LD2 / LD3 / LD4 / ST1 / ST2 / ST3 / ST4
                {
                    uint64_t datasize = 64ull << Q;
                    uint64_t esize = 8ull << size;
                    uint64_t elements = datasize / esize;
                    uint64_t selem = 1;
                    uint64_t ebytes = esize / 8;
                    uint64_t address = regs[ n ];
                    uint64_t rpt = 0;

                    if ( 2 == opcode )
                        rpt = 4;
                    else if ( 6 == opcode )
                        rpt = 3;
                    else if ( 10 == opcode )
                        rpt = 2;
                    else if ( 7 == opcode )
                        rpt = 1;
                    else if ( 8 == opcode )
                    {
                        selem = 2;
                        rpt = 1;
                    }
                    else if ( 4 == opcode )
                    {
                        selem = 3;
                        rpt = 1;
                    }
                    else if ( 0 == opcode )
                    {
                        selem = 4;
                        rpt = 1;
                    }
                    else
                        unhandled();

                    //tracer.Trace( "rpt %llu, elements %llu selem %llu, datasize %llu, esize %llu, ebytes %llu\n", rpt, elements, selem, datasize, esize, ebytes );
                    //tracer.Trace( "source data at pc %#llx:\n", pc );
                    //tracer.TraceBinaryData( getmem( address ), (uint32_t) ( rpt * elements * ebytes * selem ), 8 );
                    uint64_t offs = 0;

                    // possible bug? not differentiating (single x-element structure vs multiple structure) variants of st

                    for ( uint64_t r = 0; r < rpt; r++ ) // can't combine in one big memcpy because for rpt > 1 the registers may wrap back to 0. plus, de-interleaving
                    {
                        for ( uint64_t e = 0; e < elements; e++ )
                        {
                            uint64_t tt = ( t + r ) % 32;
                            for ( uint64_t s = 0; s < selem; s++ )
                            {
                                uint64_t eaddr = address + offs;
                                if ( L ) // LD
                                    mcpy( vreg_ptr( tt, e * ebytes ), getmem( eaddr ), ebytes );
                                else // ST
                                    mcpy( getmem( eaddr ), vreg_ptr( tt, e * ebytes ), ebytes );
                                offs += ebytes;
                                tt = ( tt + 1 ) % 32;
                            }
                        }
                    }

                    if ( L )
                        trace_vregs();

                    if ( post_index )
                    {
                        if ( 31 == m )
                        {
                            if ( 7 == opcode )
                                offs = Q ? 16 : 8;
                            else if ( 8 == opcode )
                                offs = Q ? 32 : 16;
                            else if ( 4 == opcode )
                                offs = Q ? 48 : 24;
                            else if ( 0 == opcode )
                                offs = Q ? 64 : 32;
                            else
                                unhandled();
                        }
                        else
                            offs = regs[ m ];
                        address += offs;
                        regs[ n ] = address;
                    }
                }
                else
                    unhandled();
                break;
            }
            case 0x88: // LDAXR <Wt>, [<Xn|SP>{, #0}]    ;    LDXR <Wt>, [<Xn|SP>{, #0}]    ;    STXR <Ws>, <Wt>, [<Xn|SP>{, #0}]    ;    STLXR <Ws>, <Wt>, [<Xn|SP>{, #0}]
                       //                                                                        STLR <Wt>, [<Xn|SP>{, #0}]          ;    STLR <Wt>, [<Xn|SP>, #-4]!
            case 0xc8: // LDAXR <Xt>, [<Xn|SP>{, #0}]    ;    LDXR <Xt>, [<Xn|SP>{, #0}]    ;    STXR <Ws>, <Xt>, [<Xn|SP>{, #0}]    ;    STLXR <Ws>, <Xt>, [<Xn|SP>{, #0}]
                       //                                                                        STLR <Xt>, [<Xn|SP>{, #0}]          ;    STLR <Xt>, [<Xn|SP>, #-8]!
            {
                uint64_t t = opbits( 0, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t t2 = opbits( 10, 5 );
                uint64_t s = opbits( 16, 5 );
                uint64_t L = opbits( 21, 2 );
                uint64_t bit23 = opbit( 23 );

                if ( 0x1f != t2 )
                    unhandled();

                if ( 0 == L ) // stxr, stlr
                {
                    uint64_t bit30 = opbit( 30 );
                    uint64_t tval = val_reg_or_zr( t );
                    if ( bit30 )
                        setui64( regs[ n ], tval );
                    else
                        setui32( regs[ n ], (uint32_t) tval );

                    if ( !bit23 && 31 != s ) // stxr
                        regs[ s ] = 0; // success
                }
                else if ( 2 == L ) // ldaxr or ldxr
                {
                    if ( 0x1f != s )
                        unhandled();

                    if ( 31 == t )
                        break;

                    if ( 0xc8 == hi8 )
                        regs[ t ] = getui64( regs[ n ] );
                    else
                        regs[ t ] = getui32( regs[ n ] );
                }
                break;
            }
            case 0xd6: // BLR <Xn>    ;    BR <Xn>    ;    RET {<Xn>}
            {
                uint64_t n = opbits( 5, 5 );
                uint64_t theop = opbits( 21, 2 );
                uint64_t bit23 = opbit( 23 );
                uint64_t op2 = opbits( 12, 9 );
                uint64_t A = opbit( 11 );
                uint64_t M = opbit( 10 );
                if ( bit23 || 0x1f0 != op2 || A || M )
                    unhandled();

                if ( 0 == theop || 2 == theop ) // br, ret
                    pc = regs[ n ];
                else if ( 1 == theop ) // blr
                {
                    uint64_t location = pc + 4;
                    pc = regs[ n ];
                    regs[ 30 ] = location; // hard-coded to register 30
                }
                else
                    unhandled();

                continue;
            }
            case 0x1b: // MADD <Wd>, <Wn>, <Wm>, <Wa>    ;    MSUB <Wd>, <Wn>, <Wm>, <Wa>
            case 0x9b: // MADD <Xd>, <Xn>, <Xm>, <Xa>    ;    MSUB <Xd>, <Xn>, <Xm>, <Xa>    ;    UMULH <Xd>, <Xn>, <Xm>    ;    UMADDL <Xd>, <Wn>, <Wm>, <Xa>
                       // SMADDL <Xd>, <Wn>, <Wm>, <Xa>  ;    SMULH <Xd>, <Xn>, <Xm>         ;    UMSUBL <Xd>, <Wn>, <Wm>, <Xa>
            {
                uint64_t d = opbits( 0, 5 );
                if ( 31 == d )
                    break;
                bool xregs = ( 0 != opbit( 31 ) );
                uint64_t m = opbits( 16, 5 );
                uint64_t a = opbits( 10, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t bits23_21 = opbits( 21, 3 );
                bool bit15 = ( 1 == opbit( 15 ) );
                uint64_t aval = val_reg_or_zr( a );
                uint64_t nval = val_reg_or_zr( n );
                uint64_t mval = val_reg_or_zr( m );

                if ( xregs )
                {
                    //tracer.Trace( "bits23_21 %llx, bit15 %d\n", bits23_21, bit15 );

                    if ( 0x9b == hi8 && 5 == bits23_21 && bit15 ) // UMSUBL <Xd>, <Wn>, <Wm>, <Xa>
                        regs[ d ] = regs[ a ] - ( ( regs[ n ] & 0xffffffff ) * ( regs[ m ] & 0xffffffff ) );
                    else if ( 1 == bits23_21 && bit15 ) // smsubl
                        regs[ d ] = aval - ( ( 0xffffffff & nval ) * ( 0xffffffff & mval ) );
                    else if ( 0 == bits23_21 && bit15 ) // msub
                        regs[ d ] = aval - ( nval * mval );
                    else if ( 6 == bits23_21 && 31 == a && !bit15 ) // umulh
                    {
                        uint64_t hi, lo;
                        CMultiply128:: mul_u64_u64( &hi, &lo, nval, mval );
                        regs[ d ] = hi;
                    }
                    else if ( 2 == bits23_21 && !bit15 && 31 == a ) // smulh
                    {
                        int64_t hi, lo;
                        CMultiply128:: mul_s64_s64( &hi, &lo, nval, mval );
                        regs[ d ] = hi;
                    }
                    else if ( 5 == bits23_21 && !bit15 ) // umaddl
                        regs[ d ] = aval + ( ( 0xffffffff & nval ) * ( 0xffffffff & mval ) );
                    else if ( 1 == bits23_21 && !bit15 ) // smaddl
                        regs[ d ] = aval + ( (int64_t) (int32_t) ( 0xffffffff & nval ) * (int64_t) (int32_t) ( 0xffffffff & mval ) );
                    else if ( 0 == bits23_21 && !bit15 ) // madd
                        regs[ d ] = aval + ( nval * mval );
                    else
                        unhandled();
                }
                else
                {
                    if ( 0 == bits23_21 && bit15 ) // msub
                        regs[ d ] = (uint32_t) aval - ( (uint32_t) nval * (uint32_t) mval );
                    else if ( 0 == bits23_21 && !bit15 ) // madd
                        regs[ d ] = (uint32_t) aval + ( (uint32_t) nval * (uint32_t) mval );
                    else
                        unhandled();
                }
                break;
            }
            case 0x72: // MOVK <Wd>, #<imm>{, LSL #<shift>}       ;  ANDS <Wd>, <Wn>, #<imm>
            case 0xf2: // MOVK <Xd>, #<imm>{, LSL #<shift>}       ;  ANDS <Xd>, <Xn>, #<imm>
            {
                uint64_t d = opbits( 0, 5 );
                uint64_t bit23 = opbit( 23 ); // 1 for MOVK, 0 for ANDS
                if ( bit23 )  // MOVK <Xd>, #<imm>{, LSL #<shift>}
                {
                    if ( 31 == d )
                        break;
                    uint64_t hw = opbits( 21, 2 );
                    uint64_t pos = ( hw << 4 );
                    uint64_t imm16 = opbits( 5, 16 );
                    regs[ d ] = plaster_bits( regs[ d ], imm16, pos, 16 );
                }
                else // ANDS <Xd>, <Xn>, #<imm>
                {
                    uint64_t N_immr_imms = opbits( 10, 13 );
                    uint64_t xregs = ( 0 != ( 0x80 & hi8 ) );
                    uint64_t op2 = decode_logical_immediate( N_immr_imms, xregs ? 64 : 32 );
                    uint64_t n = opbits( 5, 5 );
                    uint64_t nvalue = val_reg_or_zr( n );
                    uint64_t result = ( nvalue & op2 );
                    if ( xregs )
                        fN = get_bit( result, 63 );
                    else
                    {
                        result = (uint32_t) result;
                        fN = get_bit( result, 31 );
                    }

                    fZ = ( 0 == result );
                    fC = fV = false;
                    if ( 31 != d )
                        regs[ d ] = result;
                }
                break;
            }
            case 0x38: // B LDRB STRB
            case 0x78: // H LDRH STRH
            case 0xb8: // W
            case 0xf8: // X
            {
                // LDR <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
                // LDR <Xt>, [<Xn|SP>], #<simm>
                // LDR <Xt>, [<Xn|SP>, #<simm>]!
                // STR <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
                // STR <Xt>, [<Xn|SP>], #<simm>
                // STR <Xt>, [<Xn|SP>, #<simm>]!

                uint64_t opc = opbits( 21, 3 );
                uint64_t n = opbits( 5, 5 );
                uint64_t t = opbits( 0, 5 );

                if ( 0 == opc ) // str (immediate) post-index and pre-index
                {
                    uint64_t unsigned_imm9 = opbits( 12, 9 );
                    int64_t extended_imm9 = sign_extend( unsigned_imm9, 8 );
                    uint64_t option = opbits( 10, 2 );
                    uint64_t address = 0;

                    if ( 0 == option )
                        address = regs[ n ] + extended_imm9;
                    else if ( 1 == option )
                        address = regs[ n ];
                    else if ( 3 == option )
                    {
                        regs[ n ] += extended_imm9;
                        address = regs[ n ];
                    }
                    else
                        unhandled();

                    uint64_t val = ( 31 == t ) ? 0 : regs[ t ];

                    if ( 0x38 == hi8 )
                        setui8( address, val & 0xff );
                    else if ( 0x78 == hi8 )
                        setui16( address, val & 0xffff );
                    else if ( 0xb8 == hi8 )
                        setui32( address, (uint32_t) val );
                    else
                        setui64( address, val );

                    if ( 1 == option ) // post index
                        regs[ n ] += extended_imm9;
                }
                else if ( 1 == opc ) // str (register) STR <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
                {
                    uint64_t m = opbits( 16, 5 );
                    uint64_t shift = opbit( 12 );
                    if ( 1 == shift )
                        shift = ( hi8 >> 6 );
                    uint64_t option = opbits( 13, 3 );
                    uint64_t address = regs[ n ];
                    int64_t offset = extend_reg( m, option, shift );
                    address += offset;
                    uint64_t val = ( 31 == t ) ? 0 : regs[ t ];

                    if ( 0x38 == hi8 )
                        setui8( address, val & 0xff );
                    else if ( 0x78 == hi8 )
                        setui16( address, val & 0xffff );
                    else if ( 0xb8 == hi8 )
                        setui32( address, (uint32_t) val );
                    else
                        setui64( address, val );
                }
                else if ( 2 == opc ) // ldr (immediate)
                {
                    uint64_t unsigned_imm9 = opbits( 12, 9 );
                    int64_t extended_imm9 = sign_extend( unsigned_imm9, 8 );
                    uint64_t option = opbits( 10, 2 );
                    uint64_t address = 0;

                    if ( 0 == option )
                        address = regs[ n ] + extended_imm9;
                    else if ( 1 == option )
                        address = regs[ n ];
                    else if ( 3 == option )
                    {
                        regs[ n ] += extended_imm9;
                        address = regs[ n ];
                    }
                    else
                        unhandled();

                    if ( 0x38 == hi8 )
                        regs[ t ] = getui8( address );
                    else if ( 0x78 == hi8 )
                        regs[ t ] = getui16( address );
                    else if ( 0xb8 == hi8 )
                        regs[ t ] = getui32( address );
                    else
                        regs[ t ] = getui64( address );

                    if ( 1 == option ) // post index
                        regs[ n ] += extended_imm9;
                }
                else if ( 3 == opc ) // ldr (register) LDR <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
                {
                    uint64_t m = opbits( 16, 5 );
                    uint64_t shift = opbit( 12 );
                    if ( 1 == shift )
                        shift = ( hi8 >> 6 );

                    uint64_t option = opbits( 13, 3 );
                    uint64_t address = regs[ n ];
                    int64_t offset = extend_reg( m, option, shift );
                    address += offset;

                    if ( 0x38 == hi8 )
                        regs[ t ] = getui8( address );
                    else if ( 0x78 == hi8 )
                        regs[ t ] = getui16( address );
                    else if ( 0xb8 == hi8 )
                        regs[ t ] = getui32( address );
                    else
                        regs[ t ] = getui64( address );
                }
                else if ( 4 == opc || 6 == opc ) // LDRSW <Xt>, [<Xn|SP>], #<simm>    ;    LDURSB <Wt>, [<Xn|SP>{, #<simm>}]
                {
                    if ( 31 == t )
                        break;
                    int64_t imm9 = sign_extend( opbits( 12, 9 ), 8 );
                    uint64_t option = opbits( 10, 2 );
                    uint64_t address = 0;

                    if ( 0 == option ) // LDURSB <Wt>, [<Xn|SP>{, #<simm>}]
                        address = regs[ n ] + imm9;
                    else if ( 1 == option )
                        address = regs[ n ];
                    else if ( 3 == option )
                    {
                        regs[ n ] += imm9;
                        address = regs[ n ];
                    }
                    else
                        unhandled();

                    if ( 0x38 == hi8 )
                        regs[ t ] = sign_extend( getui8( address ), 7 );
                    else if ( 0x78 == hi8 )
                        regs[ t ] = sign_extend( getui16( address ), 15 );
                    else if ( 0xb8 == hi8 )
                        regs[ t ] = sign_extend( getui32( address ), 31 );
                    else
                        unhandled();

                    bool isx = ( 0 == opbit( 22 ) );
                    if ( !isx )
                        regs[ t ] = (uint32_t) regs[ t ];

                    if ( 1 == option ) // post index
                        regs[ n ] += imm9;
                }
                else if ( 5 == opc  || 7 == opc ) // hi8 = 0x78
                                                  //     (opc == 7)                  LDRSH <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
                                                  //     (opc == 5)                  LDRSH <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
                                                  // hi8 = 0x38
                                                  //     (opc == 7 && option != 011) LDRSB <Wt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
                                                  //     (opc == 5 && option != 011) LDRSB <Xt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
                                                  //     (opc == 7 && option == 011) LDRSB <Wt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
                                                  //     (opc == 5 && option == 011) LDRSB <Xt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
                                                  // hi8 == 0xb8
                                                  //     (opc == 5 && option = many) LDRSW <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
                {
                    uint64_t m = opbits( 16, 5 );
                    uint64_t shift = opbit( 12 );
                    if ( 1 == shift )
                        shift = ( hi8 >> 6 );
                    uint64_t option = opbits( 13, 3 );
                    bool mIsX = ( 1 == ( option & 1 ) );
                    uint64_t address = regs[ n ];
                    if ( 31 == t )
                        break;

                    if ( 0xb8 == hi8 )
                    {
                        uint64_t offset = extend_reg( m, option, opbit( 12 ) ? 2 : 0 );
                        regs[ t ] = sign_extend( getui32( address + offset ), 31 );
                    }
                    else if ( 0x38 == hi8 )
                    {
                        int64_t offset = 0;
                        if ( 3 == option )
                        {
                            uint64_t mval = val_reg_or_zr( m );
                            offset = mIsX ? mval : ( ( (uint32_t) mval ) << shift );
                        }
                        else
                            offset = extend_reg( m, option, shift );
                        address += offset;
                        regs[ t ] = sign_extend( getui8( address ), 7 );
                    }
                    else if ( 0x78 == hi8 )
                    {
                        int64_t offset = extend_reg( m, option, shift );
                        address += offset;
                        regs[ t ] = sign_extend( getui16( address ), 15 );
                    }
                    else
                        unhandled();
                }
                break;
            }
            case 0x39: // B
            case 0x79: // H                              ;    LDRSH <Wt>, [<Xn|SP>{, #<pimm>}]    ;     STR/LDR <Xt>, [<Xn|SP>{, #<pimm>}]
            case 0xb9: // W
            case 0xf9: // X ldr + str unsigned offset    ;    LDRSW <Xt>, [<Xn|SP>{, #<pimm>}]
            {
                uint64_t opc = opbits( 22, 2 );
                uint64_t imm12 = opbits( 10, 12 );
                uint64_t lsl = opbits( 30, 2 );
                imm12 <<= lsl;
                uint64_t t = opbits( 0, 5 );
                uint64_t n = opbits( 5, 5 );
                uint64_t address = regs[ n ] + imm12;

                if ( 0 == opc ) // str
                {
                    uint64_t val = val_reg_or_zr( t );

                    if ( 0x39 == hi8 )
                        setui8( address, (uint8_t) val );
                    else if ( 0x79 == hi8 )
                        setui16( address, (uint16_t) val );
                    else if ( 0xb9 == hi8 )
                        setui32( address, (uint32_t) val );
                    else
                        setui64( address, val );
                }
                else if ( 1 == opc ) // 0-extend ldr
                {
                    if ( 31 == t )
                        break;

                    if ( 0x39 == hi8 )
                        regs[ t ] = getui8( address );
                    else if ( 0x79 == hi8 )
                        regs[ t ] = getui16( address );
                    else if ( 0xb9 == hi8 )
                        regs[ t ] = getui32( address );
                    else
                        regs[ t ] = getui64( address );
                }
                else if ( 2 == opc ) // sign-extend to 64 bits ldr
                {
                    if ( 31 == t )
                        break;

                    if ( 0x39 == hi8 )
                        regs[ t ] = sign_extend( getui8( address ), 7 );
                    else if ( 0x79 == hi8 )
                        regs[ t ] = sign_extend( getui16( address ), 15 );
                    else if ( 0xb9 == hi8 )
                        regs[ t ] = sign_extend( getui32( address ), 31 );
                    else
                        unhandled();
                }
                else if ( 3 == opc ) // sign-extend to 32 bits ldr
                {
                    if ( 31 == t )
                        break;

                    if ( 0x39 == hi8 )
                        regs[ t ] = sign_extend32( getui8( address ), 7 );
                    else if ( 0x79 == hi8 )
                        regs[ t ] = sign_extend32( getui16( address ), 15 );
                    else if ( 0xb9 == hi8 )
                        regs[ t ] = getui32( address );
                    else
                        unhandled();
                }
                break;
            }
            case 0xff: // call this maximum out so the msft compiler doesn't do a bounds check at runtime for the switch jump table
            default:
                unhandled();
        }

        pc += 4;
    } //for

    return cycles;
} //run
