document.addEventListener('DOMContentLoaded', (event) => {
    var socket = io.connect(window.location.origin);
    socket.on('connect', function () {
        console.log('Connected to the server');
    });

//   socket.on('connect_error', function (error) {
//       console.log('Connect Error:', error);
////   });

//   socket.on('connect_timeout', function () {
//       console.log('Connection Timeout');
//   });

    function updateFields(data) {
        // 更新"total"字段
        document.querySelector('[data-target="total"]').innerText = data.total;
        // 更新"in_school"字段
        document.querySelector('[data-target="in_school"]').innerText = data.in_school;
        // 更新"out_school"字段
        document.querySelector('[data-target="out_school"]').innerText = data.out_school;
        // 更新"percentage"字段
        document.querySelector('[data-target="percentage"]').innerText = data.percentage + '%';
    }

    function updateTable(updateInfo) {
        var tbody = document.querySelector(".table-responsive tbody");

        // 删除旧的项目
        for (var deletedItem of updateInfo.deleted) {
            var deletedRow = document.getElementById(deletedItem.ID);
            if (deletedRow) {
                deletedRow.remove();
            }
        }

        // 添加新的项目
        for (var addedItem of updateInfo.added) {
            var row = tbody.insertRow();
            row.id = addedItem.ID;
            row.innerHTML = `<td><a class="fw-semibold">${addedItem.ID}</a></td>
                        <td>
                            <div class="d-flex gap-2 align-items-center">
                                <div class="flex-shrink-0">
                                    <img src="/static/images/users/avatar-3.jpg" alt="" class="avatar-xs rounded-circle" />
                                </div>
                                <div class="flex-grow-1">
                                    ${addedItem.Name}
                                </div>
                            </div>
                        </td>
                        <td>${addedItem.Identity}</td>
                        <td>${addedItem.Date}</td>
                        <td class="text-success"><i class="ri-checkbox-circle-line fs-17 align-middle"></i> ${addedItem.Status}</td>`;
        }
    }

    function updateChart(data) {
        console.log("Received data:", data);  // 打印接收到的数据

        // 合并现有数据和接收到的数据
        for (let key in data) {
            subjectsData[key] = data[key];
        }

        // 使用完整的 subjectsData 更新 ApexCharts 图表的数据
        chart.updateSeries([{
            name: "Scores",
            data: Object.values(subjectsData)
        }]);
        chart.updateOptions({
            xaxis: {
                categories: Object.keys(subjectsData)
            }
        });
    }

//    socket.on('video_frame', function (data) {
//        console.log("Received video_frame event");
//        var image = document.getElementById('videoFeed');
//         image.src = "data:image/jpeg;base64," + data;
//      });
    socket.on('update_table', function (data) {
        console.log('update_table event received');
        updateTable(data);
    });

//    socket.on('update_field', function (data) {
//        updateFields(data);
//    });
//     socket.on('update_chart', function (data) {
//         updateChart(data);
//     });
});