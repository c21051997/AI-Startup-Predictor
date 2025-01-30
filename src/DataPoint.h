#ifndef DATAPOINT_H
#define DATAPOINT_H

struct DataPoint {
    double funding_total_usd;
    double funding_rounds;
    double age_first_funding_year;
    double age_last_funding_year;
    double age_first_milestone_year;
    double age_last_milestone_year;
    double relationships;
    double milestones;

    double is_software;
    double is_web;
    double is_mobile;
    double is_enterprise;
    double is_advertising;
    double is_gamesvideo;
    double is_ecommerce;
    double is_biotech;
    double is_consulting;

    int is_successful;
};

#endif
