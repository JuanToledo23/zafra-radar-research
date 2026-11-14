# References

Numbered list of every paper, dataset, regulation, news item and software package cited in
this repository. Other documents cite them as [n]. Each URL was requested on 2026-10-27
(`curl -sI`, then a GET following redirects). It is marked **(verified 2026-10-27)** if it
answered 200 or 301, or if its redirect chain (for example a DOI resolver's 302) ended in
200. Otherwise it is marked **(unverified)**: the link was not replaced, it simply could not
be confirmed from a script (403 from publisher bot protection, 202 challenge pages, or a
timeout). Reference 10 is a book and has no URL.

## Prior work on satellite monitoring of sugarcane

Titles and venues of these are still to be added from the author's research notes.

1. Baghdadi et al. (2010). https://pmc.ncbi.nlm.nih.gov/articles/PMC3230945 (verified 2026-10-27)
2. Baghdadi et al. (2009). https://agritrop.cirad.fr/549410 (unverified)
3. Molijn et al. (2019). doi:10.3390/rs11091109,
   https://doi.org/10.3390/rs11091109 (unverified)
4. Stasolla & Neyt (2019). doi:10.1109/IGARSS.2019.8898706,
   https://doi.org/10.1109/IGARSS.2019.8898706 (unverified)
5. Kavats et al. (2020). doi:10.3390/rs12244080, https://doi.org/10.3390/rs12244080
   (unverified)
6. Olfindo et al. (2020). *ISPRS Annals* V-3-2020, 519.
   https://isprs-annals.copernicus.org/articles/V-3-2020/519/2020/ (verified 2026-10-27)
7. den Besten et al. (2021). doi:10.3389/frsen.2021.778691,
   https://doi.org/10.3389/frsen.2021.778691 (verified 2026-10-27)
8. den Besten et al. (2023). doi:10.1016/j.rse.2023.113555,
   https://doi.org/10.1016/j.rse.2023.113555 (verified 2026-10-27)
9. Koutilya & Bandaru (2023). doi:10.1016/j.atech.2023.100347,
   https://doi.org/10.1016/j.atech.2023.100347 (verified 2026-10-27)

## SAR statistics and change detection

10. Oliver, C. & Quegan, S. (2004). *Understanding Synthetic Aperture Radar Images.*
    SciTech. ISBN 9781891121319 (no URL; a book). Ch. 4 (speckle statistics, ENL), ch. 12
    (change detection, ratio statistics).
11. Rignot, E. J. M. & van Zyl, J. J. (1993). Change detection techniques for ERS-1 SAR
    data. *IEEE Trans. Geosci. Remote Sens.* 31(4), 896-906.
    doi:10.1109/36.239913, https://doi.org/10.1109/36.239913 (unverified)
12. Conradsen, K., Nielsen, A. A., Schou, J. & Skriver, H. (2003). A test statistic in the
    complex Wishart distribution and its application to change detection in polarimetric
    SAR data. *IEEE Trans. Geosci. Remote Sens.* 41(1), 4-19.
    doi:10.1109/TGRS.2002.808066, https://doi.org/10.1109/TGRS.2002.808066 (unverified)

## Image segmentation

13. Felzenszwalb, P. F. & Huttenlocher, D. P. (2004). Efficient graph-based image
    segmentation. *Int. J. Computer Vision* 59(2), 167-181.
    doi:10.1023/B:VISI.0000022288.19776.77,
    https://doi.org/10.1023/B:VISI.0000022288.19776.77 (verified 2026-10-27)
14. Kirillov, A. et al. (2023). Segment Anything. *Proc. ICCV*, 4015-4026.
    arXiv:2304.02643, https://arxiv.org/abs/2304.02643 (verified 2026-10-27). Code and checkpoints
    (Apache 2.0): https://github.com/facebookresearch/segment-anything (verified 2026-10-27)
15. Wu, Q. & Osco, L. (2023). samgeo: A Python package for segmenting geospatial data with
    the Segment Anything Model. *J. Open Source Software* 8(89), 5663. Package
    `segment-geospatial`. doi:10.21105/joss.05663,
    https://doi.org/10.21105/joss.05663 (verified 2026-10-27)

## Datasets and data services

16. NASA/JPL-ISRO NISAR Project. *NISAR L2 Geocoded Polarimetric Covariance (GCOV) product
    specification*, JPL D-102274, Rev F.
    https://cumulus.asf.earthdatacloud.nasa.gov/PUBLIC/DATA/NISAR/NISAR_D-102274_RevF_NASA_SDS_Product_Specification_L2_GCOV_w-esigs.pdf
    (verified 2026-10-27)
17. NASA/JPL-ISRO NISAR Project. NISAR L2 GCOV and GUNW, Provisional
    (`NISAR_L2_GCOV_PROVISIONAL_V1`, `NISAR_L2_GUNW_PROVISIONAL_V1`), distributed by ASF DAAC;
    ASF NISAR documentation: https://nisar-docs.asf.alaska.edu/ (verified 2026-10-27); provisional
    known issues: https://nisar-docs.asf.alaska.edu/provisional-known-issues/ (verified 2026-10-27);
    availability: https://nisar-docs.asf.alaska.edu/availability-overview/ (verified 2026-10-27).
    Earthdata Login: https://urs.earthdata.nasa.gov (verified 2026-10-27)
18. NASA FIRMS, Fire Information for Resource Management System:
    https://earthdata.nasa.gov/firms (verified 2026-10-27). Area API:
    https://firms.modaps.eosdis.nasa.gov/api/area/ (verified 2026-10-27) (MAP_KEY:
    https://firms.modaps.eosdis.nasa.gov/api/map_key/ (verified 2026-10-27))
19. Schroeder, W. et al. (2014). The VIIRS 375 m active fire product.
    doi:10.1016/j.rse.2013.12.008, https://doi.org/10.1016/j.rse.2013.12.008 (verified 2026-10-27)
20. ESA Copernicus Sentinel-2 MSI Level-2A, Earth Engine collection
    `COPERNICUS/S2_SR_HARMONIZED`.
    https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    (verified 2026-10-27)
21. ESA Copernicus Sentinel-1 GRD, Earth Engine collection `COPERNICUS/S1_GRD`.
    https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD (verified 2026-10-27)
22. Huffman, G. J. et al. GPM IMERG Final Precipitation L3 Half Hourly 0.1° × 0.1° V07.
    GES DISC. Earth Engine collection `NASA/GPM_L3/IMERG_V07`.
    https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V07
    (verified 2026-10-27); https://gpm.nasa.gov/data/imerg (verified 2026-10-27)
23. Zanaga, D. et al. (2022). *ESA WorldCover 10 m 2021 v200.* doi:10.5281/zenodo.7254221,
    https://doi.org/10.5281/zenodo.7254221 (verified 2026-10-27)
24. MapBiomas Project. Collection 10 of the Annual Series of Land Use and Land Cover Maps of
    Brazil. https://brasil.mapbiomas.org (verified 2026-10-27)
25. UNICA, Centre-South crushing reports: https://www.unicadata.com.br (verified 2026-10-27)
26. CONAB, cane crop surveys: https://www.conab.gov.br (verified 2026-10-27)
27. CONADESUCA, weekly zafra progress reports per sugar mill. https://www.gob.mx/conadesuca
    (verified 2026-10-27). Final production report 2022/23:
    https://www.gob.mx/cms/uploads/attachment/file/846705/Reporte_final_de_producci_n_de_ca_a_y_az_car_2023_20230801.pdf
    (verified 2026-10-27). Progress report 2023/24 (to 4 May 2024):
    https://www.gob.mx/cms/uploads/attachment/file/915315/Reporte_de_avance_de_producci_n_de_ca_a_y_az_car_Corte_al_04_de_mayo_24_.pdf
    (verified 2026-10-27). Open data: https://datamx.io/dataset/?tags=zafra (verified 2026-10-27).
    Zafra 2025/26 (exercise 10; dates and URLs in `official-docs/README.md`): weekly
    progress reports 1-36 (*Reporte de avance de producción de caña y azúcar*, cycle weeks
    7-42; gob.mx file URLs not recorded); final report, published 17 Aug 2026:
    https://www.gob.mx/cms/uploads/attachment/file/1098473/Reporte_de_cifras_finales_25_26.pdf
    (download origin of the local copy; not re-requested); KARBE weighted average of the
    last 5 zafras (2016/17-2020/21), PDF created 2021-10-28:
    https://www.gob.mx/cms/uploads/attachment/file/677981/KARBE-Ponderado_de_5__ltimas_zafras.pdf
    (download origin of the local copy; not re-requested)
28. Gorelick, N. et al. (2017). Google Earth Engine. doi:10.1016/j.rse.2017.06.031,
    https://doi.org/10.1016/j.rse.2017.06.031 (verified 2026-10-27). Used for data access to
    Sentinel-1/2, GPM, WorldCover and MapBiomas.
29. Esri World Imagery basemap (product skeleton, online mode).
    https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer (verified 2026-10-27)

## Regulation, policy and news

30. Ley de Desarrollo Sustentable de la Caña de Azúcar (LDSCA), art. 80.
    https://mley.mx/LDSCA/articulo/80/ (verified 2026-10-27); full text:
    https://www.diputados.gob.mx/LeyesBiblio/pdf/LDSCA.pdf (verified 2026-10-27). Text
    quoted below from that PDF (Cámara de Diputados, last reform DOF 14-11-2025, 45 pages;
    copy downloaded for exercise 10, see `official-docs/README.md`), verbatim from
    `pdftotext -layout` with line breaks joined. Page numbers are those printed in the PDF
    ("n de 45"); `[…]` marks omitted fractions. Spelling and typos are the law's (e.g.
    art. 75 "hasta del 10% ciento"). Art. 80 (suspension of reception or milling for
    more than 12 hours: burning orders suspended) is the article this entry was first
    listed for. The others cover burning orders, suspension of cuts and reception, and
    sanctions (*castigos*). All nine articles: `results/10/ldsca_articles.md`.

    Art. 26, pp. 13-14 (fractions II and V on p. 14):

    > Artículo 26.- Los Comités son los órganos encargados de vigilar el cumplimiento de esta Ley, y de las reglas y definiciones que acuerde el Comité Nacional y tendrán las siguientes facultades y obligaciones:
    >
    > […]
    >
    > II. Elaborar y modificar, en su caso, el programa semanal de prioridades de Corte; adecuar el programa de zafra cuando a su juicio sea conveniente y acordar la suspensión de cortes si las condiciones lo requieren;
    >
    > […]
    >
    > V. Expedir las órdenes de suspensión de riegos, de quemas y de corte, así como elaborar el acta de fin de zafra dentro de los diez días siguientes a su terminación;
    >
    > […]

    Art. 70, p. 23:

    > Artículo 70.- Por caña programada se entenderá aquella que esté comprendida dentro de las fechas de corte según el programa previo aprobado por el Comité, debidamente actualizado durante el desarrollo de la zafra y cubierta por su respectiva orden de quema en su caso, de corte y/o de suspensión de riegos, con base en su índice de madurez.

    Art. 73, p. 24:

    > Artículo 73.- Al momento de su recepción en el batey del Ingenio, la caña de azúcar deberá tener la condición y las características siguientes:
    >
    > […]
    >
    > II. Ser fresca, en el momento de su entrega, entendiéndose por ello no más de 72 horas después de su corte en el caso de caña cruda y no mas de 48 horas después de su quema;
    >
    > […]

    Art. 75, p. 24:

    > Artículo 75.- Cuando la caña rebase las 48 horas después de su quema sin ser entregada en el batey, por causas imputables al Abastecedor de Caña de azúcar, será sujeta a un castigo hasta del 10% ciento de su valor durante las primeras 24 horas posteriores siguientes y hasta el 20% de su valor durante las 24 horas posteriores, que dictaminará el Comité. La recepción de la caña con mayor tiempo de lo antes señalado, será motivo de análisis para determinar su deterioro y en su caso, el castigo correspondiente que establecerá el citado Comité.

    Art. 77, p. 24:

    > Artículo 77.- Para la organización de la cosecha en general, las cañas no programadas se sujetarán a los procedimientos siguientes:
    >
    > I. Cuando el Abastecedor de Caña de azúcar sin orden de corte y sin autorización del Comité coseche su caña, ésta podrá no ser recibida por el Ingenio, y
    >
    > II. Cuando por causa accidental una superficie con caña desarrollada se queme sin orden de corte, será castigada hasta con el 10% de su valor sin perjuicio de otros descuentos y/o castigos que le pudiera corresponder. El Comité, previa investigación de las causas que hayan provocado el accidente, determinará dicho castigo.

    Art. 80, p. 25:

    > Artículo 80.- Cuando el Ingenio por cualquier causa suspenda la recepción y/o la molienda de caña de azúcar por más de 12 horas, deberá notificar de inmediato al Comité, para que éste suspenda las órdenes de quema y reprograme los cortes, levantando el acta correspondiente de la caña quemada en campo y en trayecto al Ingenio, para su relación con la aplicación de castigos en su caso.

    Art. 81, p. 25:

    > Artículo 81.- Cuando por causas imputables al Ingenio, la caña se procese con deterioro, debido a demoras en su recepción, o bien por haber estado más de 24 horas en el batey sin molerse, se le aplicará un castigo que será hasta del 10% del valor de la caña deteriorada y hasta del 20% ciento de su valor durante las 24 horas siguientes. El importe de este castigo será determinado por el Comité y el Ingenio deberá abonarlo a la caña total aportada.

    Art. 85, pp. 26-27 (fraction VII on p. 27):

    > Artículo 85.- Se entenderán por causas imputables a los Abastecedores de Caña de azúcar de un Ingenio las siguientes:
    >
    > […]
    >
    > VII. Cuando no atiendan oportunamente la quema, el corte y el acarreo de sus cañas.

    Art. 86, p. 27:

    > Artículo 86.- En caso de cañas quemadas accidentalmente y fuera de programa, el Comité podrá llevar a cabo una reprogramación en sus frentes de corte, acudiendo en auxilio del Abastecedor de Caña afectado para facilitar la entrega, con el fin de que se industrialice el mayor volumen posible sin lesionar los intereses de otros cañeros cuyas cañas estén en proceso de corte.
31. NOM-015-SEMARNAT/AGRICULTURA-2023, use of fire on agricultural and forest land.
    https://www.ccacoalition.org/resources/official-mexican-standard-nom-015-semarnatagriculture-2023
    (unverified)
32. ProAire Morelos, air quality management programme for the state of Morelos.
    Participatory diagnosis:
    https://colaboracioncivica.org/uploads/Diagno%CC%81stico-participativo-ProAire-Morelos_compressed.pdf
    (verified 2026-10-27)
33. São Paulo state law 11.241/2002 and the 2007 agro-environmental protocol on phasing out
    pre-harvest cane burning. Law:
    https://www.al.sp.gov.br/repositorio/legislacao/lei/2002/lei-11241-19.09.2002.html
    (verified 2026-10-27); Protocolo Agroambiental: https://semil.sp.gov.br/sma/etanolverde/ (verified 2026-10-27)
34. sugaronline (2026-01-06). MEXICO: Cane harvest starts in southern Quintana Roo.
    https://www.sugaronline.com/?p=190405 (verified 2026-10-27). Title and date read from the page; the
    article body (including the 18,000 t figure) is behind a subscription and was not read.

## Software

35. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Package
    `xgboost`. doi:10.1145/2939672.2939785,
    https://doi.org/10.1145/2939672.2939785 (unverified)
36. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Package
    `scikit-learn`. *JMLR* 12.
    https://jmlr.org/papers/v12/pedregosa11a.html (verified 2026-10-27)
37. van der Walt, S. et al. (2014). scikit-image. *PeerJ*. doi:10.7717/peerj.453,
    https://doi.org/10.7717/peerj.453 (unverified). Used for
    `skimage.segmentation.felzenszwalb`.
38. Anthropic Messages API, model `claude-haiku-5-5` (per-field summary renderer).
    https://docs.claude.com (verified 2026-10-27)
