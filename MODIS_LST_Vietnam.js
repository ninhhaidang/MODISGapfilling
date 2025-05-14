// ===== 0. Thiết lập thời gian =====
var startDate = '2020-01-01';
var endDate = '2020-02-01';

var displayStart = ee.Date(startDate);
var displayEnd = displayStart.advance(2, 'day');

// ===== 1. Định nghĩa khu vực quan tâm từ shapefile Việt Nam =====
var viet_nam = ee.FeatureCollection("projects/ee-bonglantrungmuoi/assets/viet_nam");

// ===== 2. Tải dữ liệu MODIS LST =====
var modisTerra = ee.ImageCollection('MODIS/006/MOD11A1')
    .filterDate(startDate, endDate)
    .filterBounds(viet_nam)
    .select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night']);

var modisAqua = ee.ImageCollection('MODIS/006/MYD11A1')
    .filterDate(startDate, endDate)
    .filterBounds(viet_nam)
    .select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night']);

// ===== 3. Hàm lọc chất lượng =====
function filterQuality(image) {
    var qcDay = image.select('QC_Day');
    var qcNight = image.select('QC_Night');
    var qualityMaskDay = qcDay.bitwiseAnd(3).neq(3);
    var qualityMaskNight = qcNight.bitwiseAnd(3).neq(3);
    return image.addBands(image.select('LST_Day_1km').updateMask(qualityMaskDay).rename('filtered_LST_Day_1km'))
        .addBands(image.select('LST_Night_1km').updateMask(qualityMaskNight).rename('filtered_LST_Night_1km'));
}

var modisTerraFiltered = modisTerra.map(filterQuality);
var modisAquaFiltered = modisAqua.map(filterQuality);

// ===== 4. Kết hợp dữ liệu =====
var combined = modisTerraFiltered.merge(modisAquaFiltered);

// ===== 5. Xử lý riêng ban ngày và ban đêm (đổi từ K sang °C) =====
var dayData = combined.select('filtered_LST_Day_1km').map(function (image) {
    return image.multiply(0.02).subtract(273.15)
        .rename('LST_Day_1km_C')
        .set('system:time_start', image.get('system:time_start'));
});

var nightData = combined.select('filtered_LST_Night_1km').map(function (image) {
    return image.multiply(0.02).subtract(273.15)
        .rename('LST_Night_1km_C')
        .set('system:time_start', image.get('system:time_start'));
});

// ===== 6. Trung bình dài hạn =====
var meanDay = dayData.mean();
var meanNight = nightData.mean();

// ===== 7. Tính phần dư =====
var residualsDay = dayData.map(function (image) {
    return image.subtract(meanDay)
        .rename('residual_day')
        .set('system:time_start', image.get('system:time_start'));
});

var residualsNight = nightData.map(function (image) {
    return image.subtract(meanNight)
        .rename('residual_night')
        .set('system:time_start', image.get('system:time_start'));
});

// ===== 8. Làm mượt không gian =====
var smoothedResidualsDay = residualsDay.map(function (image) {
    return image.focal_mean({ radius: 3, kernelType: 'square', units: 'pixels' })
        .focal_mean({ radius: 3, kernelType: 'square', units: 'pixels' })
        .set('system:time_start', image.get('system:time_start'));
});

var smoothedResidualsNight = residualsNight.map(function (image) {
    return image.focal_mean({ radius: 3, kernelType: 'square', units: 'pixels' })
        .focal_mean({ radius: 3, kernelType: 'square', units: 'pixels' })
        .set('system:time_start', image.get('system:time_start'));
});

// ===== 9. Làm mượt thời gian =====
var temporalSmoothing = function (collection) {
    return collection.map(function (image) {
        var time = image.get('system:time_start');
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(time, null),
            image,
            function () {
                var date = ee.Date(time);
                var before = collection.filterDate(date.advance(-7, 'day'), date);
                var after = collection.filterDate(date, date.advance(7, 'day'));
                var window = before.merge(after).mean();
                return image.unmask(window)
                    .set('system:time_start', time);
            }()
        );
    });
};

var smoothedResidualsDayTemporal = temporalSmoothing(smoothedResidualsDay);
var smoothedResidualsNightTemporal = temporalSmoothing(smoothedResidualsNight);

// ===== 10. Tạo dữ liệu cuối cùng (clip theo shapefile Việt Nam) =====
var finalDay = smoothedResidualsDayTemporal.map(function (image) {
    return image.add(meanDay)
        .rename('final_LST_Day_1km_C')
        .clip(viet_nam)
        .set('system:time_start', image.get('system:time_start'));
});

var finalNight = smoothedResidualsNightTemporal.map(function (image) {
    return image.add(meanNight)
        .rename('final_LST_Night_1km_C')
        .clip(viet_nam)
        .set('system:time_start', image.get('system:time_start'));
});

// ===== 11. Hiển thị bản đồ =====
Map.centerObject(viet_nam, 6);

// Hiển thị ảnh LST final
var finalDayDisplay = finalDay.filterDate(displayStart, displayEnd);
var finalNightDisplay = finalNight.filterDate(displayStart, displayEnd);

Map.addLayer(finalDayDisplay, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Final LST Day (Gap-filled, °C)');
Map.addLayer(finalNightDisplay, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Final LST Night (Gap-filled, °C)');

// ===== 12. In thông tin kiểm tra =====
print('Final Day LST (°C):', finalDayDisplay);
print('Final Night LST (°C):', finalNightDisplay);

// ===== 13. Xuất ảnh đã gapfill cho 3 ngày cụ thể =====
// Chia Việt Nam thành 3 vùng theo latitude
var bounds = viet_nam.geometry().bounds();
var west = bounds.coordinates().get(0).get(0);
var south = bounds.coordinates().get(0).get(1);
var east = bounds.coordinates().get(0).get(2);
var north = bounds.coordinates().get(0).get(3);

// Tính toán các dải vĩ độ
var latRange = ee.Number(north).subtract(south);
var latStep = latRange.divide(3);

// Tạo danh sách các vùng
var regions = ee.List.sequence(0, 2).map(function (i) {
    var regionSouth = ee.Number(south).add(latStep.multiply(i));
    var regionNorth = regionSouth.add(latStep);
    var region = ee.Geometry.Rectangle([west, regionSouth, east, regionNorth]);
    return region;
});

// Ngày cần xuất
var exportDates = ['2020-01-01', '2020-01-15', '2020-02-01'];

// Xuất ảnh theo từng vùng và từng ngày
regions.evaluate(function (regionList) {
    regionList.forEach(function (region, regionIndex) {
        exportDates.forEach(function (date) {
            var start = ee.Date(date);
            var end = start.advance(1, 'day');

            // Lọc ảnh theo ngày
            var dayImage = finalDay.filterDate(start, end).mean();
            var nightImage = finalNight.filterDate(start, end).mean();

            // Xuất ảnh ban ngày
            Export.image.toDrive({
                image: dayImage,
                description: 'LST_Day_' + date + '_region' + (regionIndex + 1),
                folder: 'LST_Vietnam_Daily',
                region: region,
                scale: 1000,
                maxPixels: 1e13,
                crs: 'EPSG:4326'
            });

            // Xuất ảnh ban đêm
            Export.image.toDrive({
                image: nightImage,
                description: 'LST_Night_' + date + '_region' + (regionIndex + 1),
                folder: 'LST_Vietnam_Daily',
                region: region,
                scale: 1000,
                maxPixels: 1e13,
                crs: 'EPSG:4326'
            });
        });
    });
});