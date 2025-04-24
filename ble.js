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

// 11.1 Ảnh LST gốc (trước khi lọc)
var modisTerraDayOriginalRaw = modisTerra.select('LST_Day_1km')
    .filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.multiply(0.02).subtract(273.15).clip(viet_nam);
    });

var modisTerraNightOriginalRaw = modisTerra.select('LST_Night_1km')
    .filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.multiply(0.02).subtract(273.15).clip(viet_nam);
    });

var modisAquaDayOriginalRaw = modisAqua.select('LST_Day_1km')
    .filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.multiply(0.02).subtract(273.15).clip(viet_nam);
    });

var modisAquaNightOriginalRaw = modisAqua.select('LST_Night_1km')
    .filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.multiply(0.02).subtract(273.15).clip(viet_nam);
    });

Map.addLayer(modisTerraDayOriginalRaw, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'MODIS Terra Day (Raw, °C)');
Map.addLayer(modisTerraNightOriginalRaw, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'MODIS Terra Night (Raw, °C)');
Map.addLayer(modisAquaDayOriginalRaw, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'MODIS Aqua Day (Raw, °C)');
Map.addLayer(modisAquaNightOriginalRaw, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'MODIS Aqua Night (Raw, °C)');

// 11.2 Ảnh sau khi lọc QC
var modisTerraDayFiltered = modisTerraFiltered.select('filtered_LST_Day_1km')
    .filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.multiply(0.02).subtract(273.15).clip(viet_nam);
    });

var modisTerraNightFiltered = modisTerraFiltered.select('filtered_LST_Night_1km')
    .filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.multiply(0.02).subtract(273.15).clip(viet_nam);
    });

var modisAquaDayFiltered = modisAquaFiltered.select('filtered_LST_Day_1km')
    .filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.multiply(0.02).subtract(273.15).clip(viet_nam);
    });

var modisAquaNightFiltered = modisAquaFiltered.select('filtered_LST_Night_1km')
    .filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.multiply(0.02).subtract(273.15).clip(viet_nam);
    });

Map.addLayer(modisTerraDayFiltered, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'MODIS Terra Day (Filtered, °C)');
Map.addLayer(modisTerraNightFiltered, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'MODIS Terra Night (Filtered, °C)');
Map.addLayer(modisAquaDayFiltered, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'MODIS Aqua Day (Filtered, °C)');
Map.addLayer(modisAquaNightFiltered, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'MODIS Aqua Night (Filtered, °C)');

// 11.3 Ảnh mean LST
var meanDayClipped = meanDay.clip(viet_nam);
var meanNightClipped = meanNight.clip(viet_nam);

Map.addLayer(meanDayClipped, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Mean LST Day (°C)');
Map.addLayer(meanNightClipped, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Mean LST Night (°C)');

// 11.4 Ảnh residual (ảnh gốc - ảnh mean)
var residualsDayDisplay = residualsDay.filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.clip(viet_nam);
    });

var residualsNightDisplay = residualsNight.filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.clip(viet_nam);
    });

Map.addLayer(residualsDayDisplay, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Residual Day (°C)');
Map.addLayer(residualsNightDisplay, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Residual Night (°C)');

// 11.5 Ảnh residual đã làm mượt
var smoothedResidualsDayTemporalDisplay = smoothedResidualsDayTemporal.filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.clip(viet_nam);
    });

var smoothedResidualsNightTemporalDisplay = smoothedResidualsNightTemporal.filterDate(displayStart, displayEnd)
    .map(function (image) {
        return image.clip(viet_nam);
    });

Map.addLayer(smoothedResidualsDayTemporalDisplay, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Smoothed Residual Day (°C)');
Map.addLayer(smoothedResidualsNightTemporalDisplay, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Smoothed Residual Night (°C)');

// 11.6 Dữ liệu cuối cùng
var finalDayDisplay = finalDay.filterDate(displayStart, displayEnd);
var finalNightDisplay = finalNight.filterDate(displayStart, displayEnd);

Map.addLayer(finalDayDisplay, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Final LST Day (Gap-filled, °C)');
Map.addLayer(finalNightDisplay, { min: -5, max: 35, palette: ['blue', 'green', 'yellow', 'red'] }, 'Final LST Night (Gap-filled, °C)');

// ===== 12. In thông tin kiểm tra =====
print('MODIS Terra Day Original (Raw, °C):', modisTerraDayOriginalRaw);
print('MODIS Terra Night Original (Raw, °C):', modisTerraNightOriginalRaw);
print('MODIS Aqua Day Original (Raw, °C):', modisAquaDayOriginalRaw);
print('MODIS Aqua Night Original (Raw, °C):', modisAquaNightOriginalRaw);
print('MODIS Terra Day Filtered (°C):', modisTerraDayFiltered);
print('MODIS Terra Night Filtered (°C):', modisTerraNightFiltered);
print('MODIS Aqua Day Filtered (°C):', modisAquaDayFiltered);
print('MODIS Aqua Night Filtered (°C):', modisAquaNightFiltered);
print('Mean LST Day (°C):', meanDayClipped);
print('Mean LST Night (°C):', meanNightClipped);
print('Residual Day (°C):', residualsDayDisplay);
print('Residual Night (°C):', residualsNightDisplay);
print('Smoothed Residual Day (°C):', smoothedResidualsDayTemporalDisplay);
print('Smoothed Residual Night (°C):', smoothedResidualsNightTemporalDisplay);
print('Final Day LST (°C):', finalDayDisplay);
print('Final Night LST (°C):', finalNightDisplay);

// ===== 13. Xuất tất cả ảnh ra Google Drive =====
// Lưu ý: Nếu vẫn gặp lỗi "payload size exceeds limit", có thể cần chia nhỏ khu vực (region) thành các phần
// 13.1 Xuất ảnh gốc
Export.image.toDrive({
    image: modisTerraDayOriginalRaw.mean(),
    description: 'MODIS_Terra_Day_Raw_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,  // Tăng scale lên 2km để giảm kích thước
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: modisTerraNightOriginalRaw.mean(),
    description: 'MODIS_Terra_Night_Raw_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: modisAquaDayOriginalRaw.mean(),
    description: 'MODIS_Aqua_Day_Raw_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: modisAquaNightOriginalRaw.mean(),
    description: 'MODIS_Aqua_Night_Raw_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

// 13.2 Xuất ảnh sau khi lọc QC
Export.image.toDrive({
    image: modisTerraDayFiltered.mean(),
    description: 'MODIS_Terra_Day_Filtered_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: modisTerraNightFiltered.mean(),
    description: 'MODIS_Terra_Night_Filtered_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: modisAquaDayFiltered.mean(),
    description: 'MODIS_Aqua_Day_Filtered_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: modisAquaNightFiltered.mean(),
    description: 'MODIS_Aqua_Night_Filtered_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

// 13.3 Xuất ảnh mean LST
Export.image.toDrive({
    image: meanDayClipped,
    description: 'Mean_LST_Day_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: meanNightClipped,
    description: 'Mean_LST_Night_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

// 13.4 Xuất ảnh residual
Export.image.toDrive({
    image: residualsDayDisplay.mean(),
    description: 'Residual_Day_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: residualsNightDisplay.mean(),
    description: 'Residual_Night_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

// 13.5 Xuất ảnh residual đã làm mượt
Export.image.toDrive({
    image: smoothedResidualsDayTemporalDisplay.mean(),
    description: 'Smoothed_Residual_Day_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: smoothedResidualsNightTemporalDisplay.mean(),
    description: 'Smoothed_Residual_Night_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

// 13.6 Xuất dữ liệu cuối cùng
Export.image.toDrive({
    image: finalDayDisplay.mean(),
    description: 'Final_LST_Day_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});

Export.image.toDrive({
    image: finalNightDisplay.mean(),
    description: 'Final_LST_Night_2020_01',
    folder: 'LST_Vietnam',
    region: viet_nam,
    scale: 1000,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
});