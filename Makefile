.PHONY: run setup load convert export insert stop clean status view

run: setup load convert export
	@echo ""
	@echo "========================================="
	@echo "  HOÀN TẤT! File kết quả: document_database.xlsx"
	@echo "========================================="

setup:
	@echo ">>> [1/4] Pull & khởi động PostgreSQL..."
	docker compose pull
	docker compose up -d
	@echo ">>> Chờ database sẵn sàng..."
	@until docker compose exec -T postgres pg_isready -U admin -d docdb > /dev/null 2>&1; do \
		sleep 2; \
	done
	@echo ">>> Database sẵn sàng!"
	pip install -r requirements.txt -q

load:
	@echo ">>> [2/4] Load ảnh từ samples/ vào database..."
	python3 seed.py

convert:
	@echo ">>> [3/4] Convert images → OCR (Việt + Anh + LaTeX)..."
	python3 convert.py

export:
	@echo ">>> [4/4] Export database → Excel..."
	python3 convert.py export

insert:
	@echo ">>> Thêm ảnh vào database..."
	python3 convert.py insert $(IMAGES)

result:
	@echo ""
	@echo "========== KẾT QUẢ CONVERT =========="
	@docker compose exec -T postgres psql -U admin -d docdb -c "SELECT id, images, latex FROM document;"
	@echo "======================================"

view:
	@echo ""
	@echo "==================== DOCUMENT TABLE ===================="
	@docker compose exec -T postgres psql -U admin -d docdb -c "\d document"
	@echo ""
	@echo "==================== DỮ LIỆU =========================="
	@docker compose exec -T postgres psql -U admin -d docdb -c "SELECT id, CASE WHEN images LIKE 'http%' THEN '[URL] ' || LEFT(images, 50) ELSE '[LOCAL] ' || REPLACE(images, '$(PWD)/', '') END AS source, COALESCE(latex, '(chưa convert)') AS latex, created_at::timestamp(0) FROM document ORDER BY id;"
	@echo ""
	@echo "==================== THỐNG KÊ =========================="
	@docker compose exec -T postgres psql -U admin -d docdb -c "SELECT COUNT(*) AS tong, COUNT(latex) FILTER (WHERE latex IS NOT NULL AND latex != '') AS da_convert, COUNT(*) FILTER (WHERE latex IS NULL OR latex = '') AS chua_convert, COUNT(*) FILTER (WHERE images LIKE 'http%') AS tu_url, COUNT(*) FILTER (WHERE images NOT LIKE 'http%') AS tu_local FROM document;"

stop:
	docker compose down

clean:
	docker compose down -v
	rm -f document_database.xlsx

status:
	docker compose ps
	@echo "---"
	@docker compose exec -T postgres psql -U admin -d docdb -c "SELECT id, images, LEFT(latex, 60) AS latex_preview FROM document;"
