"""Shared enums for household profile, tenure, spending category, and financial zone."""

from enum import Enum, StrEnum


class FinancialZone(Enum):
    SURVIVAL = "survival"
    STABILITY = "stability"
    IMPROVEMENT = "improvement"


class Tenure(Enum):
    OWN = "OWN"
    RENT = "RENT"


class SpendingCategory(StrEnum):
    """55 CEX fusion variable codes matching fusionData/survey-processed/CEX/cat_assignment.rda.

    Lowercased to mirror the fusion.vars convention in
    fusionData/fusion/CEI/2015-2019/2019/input/CEI_2015-2019_2019_input.R.
    """

    CLOFTW = "cloftw"
    JWLBG = "jwlbg"
    EDUC = "educ"
    STDINT = "stdint"
    ELTRNP = "eltrnp"
    HOTEL = "hotel"
    OEPRD = "oeprd"
    OESRV = "oesrv"
    RECRP = "recrp"
    EATHOME = "eathome"
    EATOUT = "eatout"
    HEALTH = "health"
    FURHWR = "furhwr"
    HAPPL = "happl"
    HHPCP = "hhpcp"
    HHPCS = "hhpcs"
    HINSP = "hinsp"
    HMTIMP = "hmtimp"
    MRTGIP = "mrtgip"
    MRTGPP = "mrtgpp"
    MRTGPS = "mrtgps"
    OHOUSE = "ohouse"
    PTAXP = "ptaxp"
    RNTEXP = "rntexp"
    CHRTY = "chrty"
    FINPAY = "finpay"
    OCASH = "ocash"
    OTHINT = "othint"
    CHECK = "check"
    LIFVAL = "lifval"
    OTHDBT = "othdbt"
    OTHFIN = "othfin"
    OWNVAL = "ownval"
    RETIRE = "retire"
    RNTVAL = "rntval"
    STDDBT = "stddbt"
    STOCK = "stock"
    VEHVAL = "vehval"
    AIRSHP = "airshp"
    GAS = "gas"
    PUBTRN = "pubtrn"
    TAXIS = "taxis"
    VEHINS = "vehins"
    VEHINT = "vehint"
    VEHMLR = "vehmlr"
    VEHNEW = "vehnew"
    VEHPRD = "vehprd"
    VEHPRN = "vehprn"
    VEHREG = "vehreg"
    VEHUSD = "vehusd"
    ELEC = "elec"
    INTPHN = "intphn"
    NGAS = "ngas"
    OFUEL = "ofuel"
    WATRSH = "watrsh"
