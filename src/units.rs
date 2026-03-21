use crate::math::constants::{AMU, C, E_CHARGE, H, PI};

// ---------------------------------------------------------------------------
// Length conversion factors
// ---------------------------------------------------------------------------
const FEET_PER_METER: f64 = 3.28084;
const INCHES_PER_METER: f64 = 39.3701;
const MILES_PER_KM: f64 = 0.621371;
const METERS_PER_AU: f64 = 1.496e11;
const METERS_PER_LIGHT_YEAR: f64 = 9.461e15;
const METERS_PER_PARSEC: f64 = 3.086e16;
const METERS_PER_ANGSTROM: f64 = 1e-10;
const METERS_PER_NAUTICAL_MILE: f64 = 1852.0;

// ---------------------------------------------------------------------------
// Mass conversion factors
// ---------------------------------------------------------------------------
const LBS_PER_KG: f64 = 2.20462;
const KG_PER_SOLAR_MASS: f64 = 1.989e30;

// ---------------------------------------------------------------------------
// Energy conversion factors
// ---------------------------------------------------------------------------
const JOULES_PER_CALORIE: f64 = 4.184;
const JOULES_PER_KWH: f64 = 3.6e6;
const JOULES_PER_BTU: f64 = 1055.06;

// ---------------------------------------------------------------------------
// Pressure conversion factors
// ---------------------------------------------------------------------------
const PA_PER_ATM: f64 = 101325.0;
const PA_PER_BAR: f64 = 1e5;
const PA_PER_PSI: f64 = 6894.76;
const PA_PER_MMHG: f64 = 133.322;

// ---------------------------------------------------------------------------
// Speed conversion factors
// ---------------------------------------------------------------------------
const KMH_PER_MPS: f64 = 3.6;
const MPH_PER_MPS: f64 = 2.23694;
const KNOTS_PER_MPS: f64 = 1.94384;

// ---------------------------------------------------------------------------
// Time conversion factors
// ---------------------------------------------------------------------------
const SECONDS_PER_YEAR: f64 = 3.1557e7;

// ===========================================================================
// Length
// ===========================================================================

/// Convert meters to feet: ft = m × 3.28084
#[must_use]
pub fn meters_to_feet(m: f64) -> f64 {
    m * FEET_PER_METER
}

/// Convert feet to meters: m = ft / 3.28084
#[must_use]
pub fn feet_to_meters(ft: f64) -> f64 {
    ft / FEET_PER_METER
}

/// Convert meters to inches: in = m × 39.3701
#[must_use]
pub fn meters_to_inches(m: f64) -> f64 {
    m * INCHES_PER_METER
}

/// Convert inches to meters: m = in / 39.3701
#[must_use]
pub fn inches_to_meters(i: f64) -> f64 {
    i / INCHES_PER_METER
}

/// Convert kilometers to miles: mi = km × 0.621371
#[must_use]
pub fn km_to_miles(km: f64) -> f64 {
    km * MILES_PER_KM
}

/// Convert miles to kilometers: km = mi / 0.621371
#[must_use]
pub fn miles_to_km(mi: f64) -> f64 {
    mi / MILES_PER_KM
}

/// Convert meters to astronomical units: AU = m / 1.496×10¹¹
#[must_use]
pub fn meters_to_au(m: f64) -> f64 {
    m / METERS_PER_AU
}

/// Convert astronomical units to meters: m = AU × 1.496×10¹¹
#[must_use]
pub fn au_to_meters(au: f64) -> f64 {
    au * METERS_PER_AU
}

/// Convert meters to light-years: ly = m / 9.461×10¹⁵
#[must_use]
pub fn meters_to_light_years(m: f64) -> f64 {
    m / METERS_PER_LIGHT_YEAR
}

/// Convert light-years to meters: m = ly × 9.461×10¹⁵
#[must_use]
pub fn light_years_to_meters(ly: f64) -> f64 {
    ly * METERS_PER_LIGHT_YEAR
}

/// Convert meters to parsecs: pc = m / 3.086×10¹⁶
#[must_use]
pub fn meters_to_parsec(m: f64) -> f64 {
    m / METERS_PER_PARSEC
}

/// Convert parsecs to meters: m = pc × 3.086×10¹⁶
#[must_use]
pub fn parsec_to_meters(pc: f64) -> f64 {
    pc * METERS_PER_PARSEC
}

/// Convert angstroms to meters: m = Å × 10⁻¹⁰
#[must_use]
pub fn angstrom_to_meters(a: f64) -> f64 {
    a * METERS_PER_ANGSTROM
}

/// Convert meters to angstroms: Å = m / 10⁻¹⁰
#[must_use]
pub fn meters_to_angstrom(m: f64) -> f64 {
    m / METERS_PER_ANGSTROM
}

/// Convert nautical miles to meters: m = nmi × 1852
#[must_use]
pub fn nautical_miles_to_meters(nm: f64) -> f64 {
    nm * METERS_PER_NAUTICAL_MILE
}

/// Convert meters to nautical miles: nmi = m / 1852
#[must_use]
pub fn meters_to_nautical_miles(m: f64) -> f64 {
    m / METERS_PER_NAUTICAL_MILE
}

// ===========================================================================
// Mass
// ===========================================================================

/// Convert kilograms to pounds: lb = kg × 2.20462
#[must_use]
pub fn kg_to_lbs(kg: f64) -> f64 {
    kg * LBS_PER_KG
}

/// Convert pounds to kilograms: kg = lb / 2.20462
#[must_use]
pub fn lbs_to_kg(lbs: f64) -> f64 {
    lbs / LBS_PER_KG
}

/// Convert kilograms to solar masses: M☉ = kg / 1.989×10³⁰
#[must_use]
pub fn kg_to_solar_masses(kg: f64) -> f64 {
    kg / KG_PER_SOLAR_MASS
}

/// Convert solar masses to kilograms: kg = M☉ × 1.989×10³⁰
#[must_use]
pub fn solar_masses_to_kg(sm: f64) -> f64 {
    sm * KG_PER_SOLAR_MASS
}

/// Convert atomic mass units to kilograms: kg = amu × 1.66054×10⁻²⁷
#[must_use]
pub fn amu_to_kg(amu: f64) -> f64 {
    amu * AMU
}

/// Convert kilograms to atomic mass units: amu = kg / 1.66054×10⁻²⁷
#[must_use]
pub fn kg_to_amu(kg: f64) -> f64 {
    kg / AMU
}

// ===========================================================================
// Energy
// ===========================================================================

/// Convert joules to electron-volts: eV = J / e
#[must_use]
pub fn joules_to_ev(j: f64) -> f64 {
    j / E_CHARGE
}

/// Convert electron-volts to joules: J = eV × e
#[must_use]
pub fn ev_to_joules(ev: f64) -> f64 {
    ev * E_CHARGE
}

/// Convert joules to calories: cal = J / 4.184
#[must_use]
pub fn joules_to_calories(j: f64) -> f64 {
    j / JOULES_PER_CALORIE
}

/// Convert calories to joules: J = cal × 4.184
#[must_use]
pub fn calories_to_joules(cal: f64) -> f64 {
    cal * JOULES_PER_CALORIE
}

/// Convert joules to kilowatt-hours: kWh = J / 3.6×10⁶
#[must_use]
pub fn joules_to_kwh(j: f64) -> f64 {
    j / JOULES_PER_KWH
}

/// Convert kilowatt-hours to joules: J = kWh × 3.6×10⁶
#[must_use]
pub fn kwh_to_joules(kwh: f64) -> f64 {
    kwh * JOULES_PER_KWH
}

/// Convert joules to British thermal units: BTU = J / 1055.06
#[must_use]
pub fn joules_to_btu(j: f64) -> f64 {
    j / JOULES_PER_BTU
}

/// Convert British thermal units to joules: J = BTU × 1055.06
#[must_use]
pub fn btu_to_joules(btu: f64) -> f64 {
    btu * JOULES_PER_BTU
}

/// Converts photon energy in eV to wavelength in meters via λ = hc/E.
#[must_use]
pub fn ev_to_wavelength(ev: f64) -> f64 {
    assert!(ev > 0.0, "energy in eV must be positive");
    let energy_joules = ev * E_CHARGE;
    (H * C) / energy_joules
}

/// Converts photon wavelength in meters to energy in eV via E = hc/λ.
#[must_use]
pub fn wavelength_to_ev(wavelength: f64) -> f64 {
    assert!(wavelength > 0.0, "wavelength must be positive");
    let energy_joules = (H * C) / wavelength;
    energy_joules / E_CHARGE
}

// ===========================================================================
// Pressure
// ===========================================================================

/// Convert pascals to atmospheres: atm = Pa / 101325
#[must_use]
pub fn pa_to_atm(pa: f64) -> f64 {
    pa / PA_PER_ATM
}

/// Convert atmospheres to pascals: Pa = atm × 101325
#[must_use]
pub fn atm_to_pa(atm: f64) -> f64 {
    atm * PA_PER_ATM
}

/// Convert pascals to bar: bar = Pa / 10⁵
#[must_use]
pub fn pa_to_bar(pa: f64) -> f64 {
    pa / PA_PER_BAR
}

/// Convert bar to pascals: Pa = bar × 10⁵
#[must_use]
pub fn bar_to_pa(bar: f64) -> f64 {
    bar * PA_PER_BAR
}

/// Convert pascals to pounds per square inch: psi = Pa / 6894.76
#[must_use]
pub fn pa_to_psi(pa: f64) -> f64 {
    pa / PA_PER_PSI
}

/// Convert pounds per square inch to pascals: Pa = psi × 6894.76
#[must_use]
pub fn psi_to_pa(psi: f64) -> f64 {
    psi * PA_PER_PSI
}

/// Convert pascals to millimeters of mercury: mmHg = Pa / 133.322
#[must_use]
pub fn pa_to_mmhg(pa: f64) -> f64 {
    pa / PA_PER_MMHG
}

/// Convert millimeters of mercury to pascals: Pa = mmHg × 133.322
#[must_use]
pub fn mmhg_to_pa(mmhg: f64) -> f64 {
    mmhg * PA_PER_MMHG
}

// ===========================================================================
// Angle
// ===========================================================================

/// Convert degrees to radians: rad = deg × π / 180
#[must_use]
pub fn degrees_to_radians(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Convert radians to degrees: deg = rad × 180 / π
#[must_use]
pub fn radians_to_degrees(rad: f64) -> f64 {
    rad * 180.0 / PI
}

/// Convert revolutions per minute to radians per second: ω = rpm × 2π / 60
#[must_use]
pub fn rpm_to_rad_per_sec(rpm: f64) -> f64 {
    rpm * 2.0 * PI / 60.0
}

/// Convert radians per second to revolutions per minute: rpm = ω × 60 / (2π)
#[must_use]
pub fn rad_per_sec_to_rpm(omega: f64) -> f64 {
    omega * 60.0 / (2.0 * PI)
}

// ===========================================================================
// Time
// ===========================================================================

/// Convert seconds to years: yr = s / 3.1557×10⁷
#[must_use]
pub fn seconds_to_years(s: f64) -> f64 {
    s / SECONDS_PER_YEAR
}

/// Convert years to seconds: s = yr × 3.1557×10⁷
#[must_use]
pub fn years_to_seconds(yr: f64) -> f64 {
    yr * SECONDS_PER_YEAR
}

// ===========================================================================
// Speed
// ===========================================================================

/// Convert meters per second to kilometers per hour: km/h = m/s × 3.6
#[must_use]
pub fn mps_to_kmh(mps: f64) -> f64 {
    mps * KMH_PER_MPS
}

/// Convert kilometers per hour to meters per second: m/s = km/h / 3.6
#[must_use]
pub fn kmh_to_mps(kmh: f64) -> f64 {
    kmh / KMH_PER_MPS
}

/// Convert meters per second to miles per hour: mph = m/s × 2.23694
#[must_use]
pub fn mps_to_mph(mps: f64) -> f64 {
    mps * MPH_PER_MPS
}

/// Convert miles per hour to meters per second: m/s = mph / 2.23694
#[must_use]
pub fn mph_to_mps(mph: f64) -> f64 {
    mph / MPH_PER_MPS
}

/// Convert meters per second to knots: kt = m/s × 1.94384
#[must_use]
pub fn mps_to_knots(mps: f64) -> f64 {
    mps * KNOTS_PER_MPS
}

/// Convert knots to meters per second: m/s = kt / 1.94384
#[must_use]
pub fn knots_to_mps(kt: f64) -> f64 {
    kt / KNOTS_PER_MPS
}

/// Convert meters per second to Mach number: M = v / v_sound
#[must_use]
pub fn mps_to_mach(mps: f64, speed_of_sound: f64) -> f64 {
    assert!(speed_of_sound > 0.0, "speed_of_sound must be positive");
    mps / speed_of_sound
}

/// Convert Mach number to meters per second: v = M × v_sound
#[must_use]
pub fn mach_to_mps(mach: f64, speed_of_sound: f64) -> f64 {
    mach * speed_of_sound
}

// ---------------------------------------------------------------------------
// Power conversions
// ---------------------------------------------------------------------------

const WATTS_PER_HP: f64 = 745.7;
const WATTS_PER_BTU_HR: f64 = 0.293_071;
const WATTS_PER_TON_REFRIG: f64 = 3516.85;
const WATTS_PER_KCAL_HR: f64 = 1.163;

/// Convert watts to mechanical horsepower: hp = W / 745.7
#[must_use] pub fn watts_to_horsepower(w: f64) -> f64 { w / WATTS_PER_HP }
/// Convert mechanical horsepower to watts: W = hp × 745.7
#[must_use] pub fn horsepower_to_watts(hp: f64) -> f64 { hp * WATTS_PER_HP }
/// Convert watts to BTU per hour: BTU/h = W / 0.293071
#[must_use] pub fn watts_to_btu_per_hour(w: f64) -> f64 { w / WATTS_PER_BTU_HR }
/// Convert BTU per hour to watts: W = BTU/h × 0.293071
#[must_use] pub fn btu_per_hour_to_watts(btu_hr: f64) -> f64 { btu_hr * WATTS_PER_BTU_HR }
/// Convert watts to tons of refrigeration: TR = W / 3516.85
#[must_use] pub fn watts_to_tons_refrigeration(w: f64) -> f64 { w / WATTS_PER_TON_REFRIG }
/// Convert tons of refrigeration to watts: W = TR × 3516.85
#[must_use] pub fn tons_refrigeration_to_watts(tons: f64) -> f64 { tons * WATTS_PER_TON_REFRIG }
/// Convert watts to kilocalories per hour: kcal/h = W / 1.163
#[must_use] pub fn watts_to_kcal_per_hour(w: f64) -> f64 { w / WATTS_PER_KCAL_HR }
/// Convert kilocalories per hour to watts: W = kcal/h × 1.163
#[must_use] pub fn kcal_per_hour_to_watts(kcal_hr: f64) -> f64 { kcal_hr * WATTS_PER_KCAL_HR }
/// Convert kilowatts to watts: W = kW × 10³
#[must_use] pub fn kilowatts_to_watts(kw: f64) -> f64 { kw * 1e3 }
/// Convert watts to kilowatts: kW = W × 10⁻³
#[must_use] pub fn watts_to_kilowatts(w: f64) -> f64 { w * 1e-3 }
/// Convert megawatts to watts: W = MW × 10⁶
#[must_use] pub fn megawatts_to_watts(mw: f64) -> f64 { mw * 1e6 }
/// Convert watts to megawatts: MW = W × 10⁻⁶
#[must_use] pub fn watts_to_megawatts(w: f64) -> f64 { w * 1e-6 }

// ---------------------------------------------------------------------------
// Electrical energy
// ---------------------------------------------------------------------------

/// Convert watt-hours to joules: J = Wh × 3600
#[must_use] pub fn watt_hours_to_joules(wh: f64) -> f64 { wh * 3600.0 }
/// Convert joules to watt-hours: Wh = J / 3600
#[must_use] pub fn joules_to_watt_hours(j: f64) -> f64 { j / 3600.0 }
/// Convert ampere-hours to coulombs: C = Ah × 3600
#[must_use] pub fn amp_hours_to_coulombs(ah: f64) -> f64 { ah * 3600.0 }
/// Convert coulombs to ampere-hours: Ah = C / 3600
#[must_use] pub fn coulombs_to_amp_hours(c: f64) -> f64 { c / 3600.0 }

// ---------------------------------------------------------------------------
// Fuel / energy equivalents
// ---------------------------------------------------------------------------

const JOULES_PER_KG_TNT: f64 = 4.184e6;
const JOULES_PER_KG_COAL: f64 = 2.9e7;
const JOULES_PER_KG_OIL: f64 = 4.187e7;
const JOULES_PER_KG_HYDROGEN: f64 = 1.42e8;
const JOULES_PER_LITER_GASOLINE: f64 = 3.4e7;

/// Convert kg of TNT equivalent to joules: J = kg × 4.184×10⁶
#[must_use] pub fn kg_tnt_to_joules(kg: f64) -> f64 { kg * JOULES_PER_KG_TNT }
/// Convert joules to kg of TNT equivalent: kg = J / 4.184×10⁶
#[must_use] pub fn joules_to_kg_tnt(j: f64) -> f64 { j / JOULES_PER_KG_TNT }
/// Convert kg of coal equivalent to joules: J = kg × 2.9×10⁷
#[must_use] pub fn kg_coal_to_joules(kg: f64) -> f64 { kg * JOULES_PER_KG_COAL }
/// Convert joules to kg of coal equivalent: kg = J / 2.9×10⁷
#[must_use] pub fn joules_to_kg_coal(j: f64) -> f64 { j / JOULES_PER_KG_COAL }
/// Convert kg of oil equivalent to joules: J = kg × 4.187×10⁷
#[must_use] pub fn kg_oil_to_joules(kg: f64) -> f64 { kg * JOULES_PER_KG_OIL }
/// Convert joules to kg of oil equivalent: kg = J / 4.187×10⁷
#[must_use] pub fn joules_to_kg_oil(j: f64) -> f64 { j / JOULES_PER_KG_OIL }
/// Convert kg of hydrogen to joules: J = kg × 1.42×10⁸
#[must_use] pub fn kg_hydrogen_to_joules(kg: f64) -> f64 { kg * JOULES_PER_KG_HYDROGEN }
/// Convert joules to kg of hydrogen equivalent: kg = J / 1.42×10⁸
#[must_use] pub fn joules_to_kg_hydrogen(j: f64) -> f64 { j / JOULES_PER_KG_HYDROGEN }
/// Convert liters of gasoline to joules: J = L × 3.4×10⁷
#[must_use] pub fn liters_gasoline_to_joules(liters: f64) -> f64 { liters * JOULES_PER_LITER_GASOLINE }
/// Convert joules to liters of gasoline equivalent: L = J / 3.4×10⁷
#[must_use] pub fn joules_to_liters_gasoline(j: f64) -> f64 { j / JOULES_PER_LITER_GASOLINE }

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-4;

    fn approx(a: f64, b: f64) -> bool {
        if b == 0.0 {
            return a.abs() < TOLERANCE;
        }
        ((a - b) / b).abs() < TOLERANCE
    }

    // -- Length roundtrips --------------------------------------------------

    #[test]
    fn roundtrip_meters_feet() {
        let m = 100.0;
        assert!(approx(feet_to_meters(meters_to_feet(m)), m));
    }

    #[test]
    fn roundtrip_meters_inches() {
        let m = 2.5;
        assert!(approx(inches_to_meters(meters_to_inches(m)), m));
    }

    #[test]
    fn roundtrip_km_miles() {
        let km = 42.195;
        assert!(approx(miles_to_km(km_to_miles(km)), km));
    }

    #[test]
    fn roundtrip_meters_au() {
        let m = 1.496e11;
        assert!(approx(au_to_meters(meters_to_au(m)), m));
    }

    #[test]
    fn roundtrip_meters_light_years() {
        let m = 9.461e15;
        assert!(approx(light_years_to_meters(meters_to_light_years(m)), m));
    }

    #[test]
    fn roundtrip_meters_parsec() {
        let m = 3.086e16;
        assert!(approx(parsec_to_meters(meters_to_parsec(m)), m));
    }

    #[test]
    fn roundtrip_angstrom_meters() {
        let a = 5.0;
        assert!(approx(meters_to_angstrom(angstrom_to_meters(a)), a));
    }

    #[test]
    fn roundtrip_nautical_miles_meters() {
        let nm = 10.0;
        assert!(approx(meters_to_nautical_miles(nautical_miles_to_meters(nm)), nm));
    }

    // -- Length spot checks -------------------------------------------------

    #[test]
    fn one_meter_in_feet() {
        assert!(approx(meters_to_feet(1.0), 3.28084));
    }

    #[test]
    fn one_km_in_miles() {
        assert!(approx(km_to_miles(1.0), 0.621371));
    }

    // -- Mass roundtrips ----------------------------------------------------

    #[test]
    fn roundtrip_kg_lbs() {
        let kg = 75.0;
        assert!(approx(lbs_to_kg(kg_to_lbs(kg)), kg));
    }

    #[test]
    fn roundtrip_kg_solar_masses() {
        let kg = 1.989e30;
        assert!(approx(solar_masses_to_kg(kg_to_solar_masses(kg)), kg));
    }

    #[test]
    fn roundtrip_amu_kg() {
        let amu_val = 12.0;
        assert!(approx(kg_to_amu(amu_to_kg(amu_val)), amu_val));
    }

    // -- Energy roundtrips --------------------------------------------------

    #[test]
    fn roundtrip_joules_ev() {
        let j = 1.602e-19;
        assert!(approx(ev_to_joules(joules_to_ev(j)), j));
    }

    #[test]
    fn roundtrip_joules_calories() {
        let j = 4184.0;
        assert!(approx(calories_to_joules(joules_to_calories(j)), j));
    }

    #[test]
    fn roundtrip_joules_kwh() {
        let j = 3.6e6;
        assert!(approx(kwh_to_joules(joules_to_kwh(j)), j));
    }

    #[test]
    fn roundtrip_joules_btu() {
        let j = 1055.06;
        assert!(approx(btu_to_joules(joules_to_btu(j)), j));
    }

    #[test]
    fn roundtrip_ev_wavelength() {
        let ev = 2.0;
        assert!(approx(wavelength_to_ev(ev_to_wavelength(ev)), ev));
    }

    // -- Pressure roundtrips ------------------------------------------------

    #[test]
    fn roundtrip_pa_atm() {
        let pa = 101325.0;
        assert!(approx(atm_to_pa(pa_to_atm(pa)), pa));
    }

    #[test]
    fn roundtrip_pa_bar() {
        let pa = 1e5;
        assert!(approx(bar_to_pa(pa_to_bar(pa)), pa));
    }

    #[test]
    fn roundtrip_pa_psi() {
        let pa = 6894.76;
        assert!(approx(psi_to_pa(pa_to_psi(pa)), pa));
    }

    #[test]
    fn roundtrip_pa_mmhg() {
        let pa = 133.322;
        assert!(approx(mmhg_to_pa(pa_to_mmhg(pa)), pa));
    }

    // -- Angle roundtrips ---------------------------------------------------

    #[test]
    fn roundtrip_degrees_radians() {
        let deg = 180.0;
        assert!(approx(radians_to_degrees(degrees_to_radians(deg)), deg));
    }

    #[test]
    fn ninety_degrees_in_radians() {
        assert!(approx(degrees_to_radians(90.0), PI / 2.0));
    }

    #[test]
    fn roundtrip_rpm_rad_per_sec() {
        let rpm = 60.0;
        assert!(approx(rad_per_sec_to_rpm(rpm_to_rad_per_sec(rpm)), rpm));
    }

    #[test]
    fn one_rpm_is_2pi_over_60() {
        assert!(approx(rpm_to_rad_per_sec(1.0), 2.0 * PI / 60.0));
    }

    // -- Time roundtrips ----------------------------------------------------

    #[test]
    fn roundtrip_seconds_years() {
        let s = 3.1557e7;
        assert!(approx(years_to_seconds(seconds_to_years(s)), s));
    }

    // -- Speed roundtrips ---------------------------------------------------

    #[test]
    fn roundtrip_mps_kmh() {
        let mps = 10.0;
        assert!(approx(kmh_to_mps(mps_to_kmh(mps)), mps));
    }

    #[test]
    fn roundtrip_mps_mph() {
        let mps = 10.0;
        assert!(approx(mph_to_mps(mps_to_mph(mps)), mps));
    }

    #[test]
    fn roundtrip_mps_knots() {
        let mps = 10.0;
        assert!(approx(knots_to_mps(mps_to_knots(mps)), mps));
    }

    #[test]
    fn roundtrip_mps_mach() {
        let mps = 680.0;
        let sos = 343.0;
        assert!(approx(mach_to_mps(mps_to_mach(mps, sos), sos), mps));
    }

    #[test]
    fn ten_mps_in_kmh() {
        assert!(approx(mps_to_kmh(10.0), 36.0));
    }

    // -- Power roundtrips ---------------------------------------------------

    #[test]
    fn roundtrip_watts_horsepower() {
        let w = 1000.0;
        assert!(approx(horsepower_to_watts(watts_to_horsepower(w)), w));
    }

    #[test]
    fn one_hp_in_watts() {
        assert!(approx(horsepower_to_watts(1.0), 745.7));
    }

    #[test]
    fn roundtrip_watts_btu_hr() {
        let w = 5000.0;
        assert!(approx(btu_per_hour_to_watts(watts_to_btu_per_hour(w)), w));
    }

    #[test]
    fn roundtrip_watts_tons_refrig() {
        let w = 3516.85;
        assert!(approx(tons_refrigeration_to_watts(watts_to_tons_refrigeration(w)), w));
    }

    #[test]
    fn roundtrip_watts_kcal_hr() {
        let w = 1000.0;
        assert!(approx(kcal_per_hour_to_watts(watts_to_kcal_per_hour(w)), w));
    }

    #[test]
    fn kw_mw_conversions() {
        assert!(approx(kilowatts_to_watts(1.0), 1000.0));
        assert!(approx(megawatts_to_watts(1.0), 1e6));
        assert!(approx(watts_to_kilowatts(1000.0), 1.0));
        assert!(approx(watts_to_megawatts(1e6), 1.0));
    }

    // -- Electrical energy --------------------------------------------------

    #[test]
    fn roundtrip_watt_hours_joules() {
        let wh = 100.0;
        assert!(approx(joules_to_watt_hours(watt_hours_to_joules(wh)), wh));
    }

    #[test]
    fn one_wh_in_joules() {
        assert!(approx(watt_hours_to_joules(1.0), 3600.0));
    }

    #[test]
    fn roundtrip_amp_hours_coulombs() {
        let ah = 5.0;
        assert!(approx(coulombs_to_amp_hours(amp_hours_to_coulombs(ah)), ah));
    }

    // -- Fuel equivalents ---------------------------------------------------

    #[test]
    fn roundtrip_kg_tnt() {
        let kg = 10.0;
        assert!(approx(joules_to_kg_tnt(kg_tnt_to_joules(kg)), kg));
    }

    #[test]
    fn roundtrip_kg_coal() {
        let kg = 1.0;
        assert!(approx(joules_to_kg_coal(kg_coal_to_joules(kg)), kg));
    }

    #[test]
    fn roundtrip_liters_gasoline() {
        let l = 50.0;
        assert!(approx(joules_to_liters_gasoline(liters_gasoline_to_joules(l)), l));
    }

    #[test]
    fn roundtrip_kg_oil() {
        let kg = 2.5;
        assert!(approx(joules_to_kg_oil(kg_oil_to_joules(kg)), kg));
    }

    #[test]
    fn roundtrip_kg_hydrogen() {
        let kg = 0.5;
        assert!(approx(joules_to_kg_hydrogen(kg_hydrogen_to_joules(kg)), kg));
    }

    #[test]
    fn test_approx_zero_b() {
        assert!(approx(0.0, 0.0));
        assert!(!approx(1.0, 0.0));
    }
}
