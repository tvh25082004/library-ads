.PHONY: run setup seed convert insert stop clean status

run: setup seed convert result

setup:
	@echo ">>> Pull & khởi động PostgreSQL..."
	docker compose pull
	docker compose up -d
	@echo ">>> Chờ database sẵn sàng..."
	@until docker compose exec -T postgres pg_isready -U admin -d docdb > /dev/null 2>&1; do \
		sleep 2; \
	done
	@echo ">>> Database sẵn sàng!"
	pip install -r requirements.txt -q

seed:
	@echo ">>> Tạo dữ liệu mẫu (ảnh công thức toán)..."
	python3 seed.py

convert:
	@echo ">>> Convert images -> LaTeX..."
	python3 convert.py

insert:
	@echo ">>> Thêm ảnh vào database..."
	python3 convert.py insert $(IMAGES)

result:
	@echo ""
	@echo "========== KẾT QUẢ CONVERT =========="
	@docker compose exec -T postgres psql -U admin -d docdb -c "SELECT id, images, latex FROM document;"
	@echo "======================================"

stop:
	docker compose down

clean:
	docker compose down -v

status:
	docker compose ps
	@echo "---"
	@docker compose exec -T postgres psql -U admin -d docdb -c "SELECT id, images, LEFT(latex, 60) AS latex_preview FROM document;"
